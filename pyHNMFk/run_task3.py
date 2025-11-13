# %%
import base64
import io
from pathlib import Path
import pickle
import functools
import itertools

import jax.numpy as jnp
import jax
jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_debug_key_reuse", True)

# set cache size to 1GB
jax.config.update("jax_compilation_cache_max_size", 2**30 - 1)

# jax.config.update("jax_log_compiles", True)
# jax.config.update("jax_compiler_detailed_logging_min_ops", 50)

# jax.config.update("jax_explain_cache_misses", True)
# jax.config.update("jax_check_tracer_leaks", True)
# jax.config.update("jax_debug_nans", True)

import orthax

import matplotlib
import matplotlib.pyplot as plt
# import matplotlib.animation as animation
import pandas as pd
from scipy.io import loadmat, savemat
from scipy.stats import ranksums

# import plotly.graph_objects as go

from hnmf_tr_optimizer.hnmf_optimizer import HNMFOptimizer, NewHNMFOptimizer, PerturbanceHNMFOptimizer, RedoHNMFOptimizer
from hnmf_tr_optimizer.clusts import result_analysis

from dls_model import InitParamsGenerator2
# plt.style.use('Solarize_Light2')

# from IPython.display import Markdown, display



# %% [markdown]
# ## models, optimizers, helpers, etc

# %%
# Plotting theme setup


TEXT_COLOR = "white"
BG_COLOR = "black"

plt.rcParams["axes.facecolor"] = plt.rcParams["figure.facecolor"] = BG_COLOR
plt.rcParams["text.color"] = TEXT_COLOR
plt.rcParams["axes.labelcolor"] = TEXT_COLOR
plt.rcParams["xtick.color"] = TEXT_COLOR
plt.rcParams["ytick.color"] = TEXT_COLOR
plt.rcParams.update({
	"axes.grid" : True,
	"grid.color": "green",
	"grid.alpha": 0.35,
	"grid.linestyle": (0, (10, 10)),
})

# BETTER SIZES
DEFAULT_W, DEFAULT_H = (16, 9)
plt.rcParams["figure.figsize"] = [DEFAULT_W, DEFAULT_H]
plt.rcParams["font.size"] = 14
plt.rcParams["figure.dpi"] = 90

plt.style.use('dark_background')


# %%
def scatter_vector(theta, lambda_0=633e-9, n=1.33, radians=False):
    """
    theta - scatter angle
    lambda_0 - lsder wavelength in meters
    n - refractive index of water
    """
    if not radians:
        theta = jnp.radians(theta)
    return (4 * jnp.pi * n/lambda_0) * jnp.sin(theta / 2)

def diffusion_coef(r, k_B=1.38e-23, T=298.15, eta=0.00089):
    """
    r - particle radius
    k_b - Boltzmann constant (J/K)
    T - Temperature (K)
    eta - Viscosity of water at room temerature (Pa*s)
    """
    return k_B * T / (6 * jnp.pi * eta * r)

def prep_data(datafile):
    df = pd.read_csv(datafile, delimiter='\t', header=None)
    df = df.iloc[1:] # remove strange first point
    d = jnp.array(df.to_numpy())

    t = d[:, 0]
    t *=  1e-3 # convert timestamps from ms to s

    # observation data is g2(t) - 1
    g2_minus1_obs = d[:, 1:].T

    div = jax.vmap(lambda x: x/x[0])
    g1_squared = div(g2_minus1_obs) # normalization by first term handles removing the beta term (roughly)

    # g1 = jnp.sqrt(jnp.maximum(g1_squared, 0)) # this line converts to g1

    g1 = jnp.where(
        jnp.greater_equal(g1_squared, 0),
        jnp.sqrt(g1_squared),
        -jnp.sqrt(-g1_squared)
    )

    # g1 = jnp.sign(g1_squared) * jnp.sqrt(jnp.abs(g1_squared))

    theta = jnp.arange(30., 151, 5) # angles known in advance - in degrees
    q = scatter_vector(theta)
    return q, t, g1

def prep_data_g2(datafile):
    df = pd.read_csv(datafile, delimiter='\t', header=None)
    df = df.iloc[1:] # remove strange first point
    d = jnp.array(df.to_numpy())

    t = d[:, 0]
    t *=  1e-3 # convert timestamps from ms to s

    # observation data is g2(t) - 1
    g2_minus1_obs = d[:, 1:].T

    # div = jax.vmap(lambda x: x/x[0])
    # g1_squared = div(g2_minus1_obs) # normalization by first term handles removing the beta term (roughly)

    # # g1 = jnp.sqrt(jnp.maximum(g1_squared, 0)) # this line converts to g1
    # g1 = jnp.where(
    #     jnp.greater_equal(g1_squared, 0),
    #     jnp.sqrt(g1_squared),
    #     -jnp.sqrt(-g1_squared)
    # )

    theta = jnp.arange(30., 151, 5) # angles known in advance - in degrees
    q = scatter_vector(theta)
    return q, t, g2_minus1_obs

# for drawing normal curves
def normal_distribution_single(x, amplitude, mu, sigma):
    return amplitude * jnp.exp(-(x-mu)**2/(2*sigma**2))/jnp.sqrt(2*jnp.pi*sigma**2)

normal_distributions = jax.vmap(normal_distribution_single, in_axes=(None, 0, 0, 0))

def normal_distribution(possible_D, amp, mu, sig):
    whole = normal_distributions(possible_D, amp, mu, sig).sum(axis=0)
    return whole / jnp.sum(whole) # normalize for plotting


# %%
### models

SCALING_CONST = 2.45e-7


######### Normal model ##########

def get_g1(t, nk, a, c):
    b = a**2/2
    d = jnp.sqrt(2)
    s_pi = jnp.sqrt(jnp.pi)
    x = (-a + c*t)/d
    y = jnp.where(
        jnp.greater_equal(x, 5),
        1/(x*s_pi),
        jax.scipy.special.erfc(x)*jnp.exp(x**2)
    )
    e = (nk/2)*jnp.exp(-b)
    return e*y

# vectorize along time dimension
all_g1 = jax.vmap(
    get_g1,
    in_axes=(0, None, None, None)
)

def source3_1(t, q, amp, mu, sig):
    ################
    const = SCALING_CONST
    # const = 1.0
    ################
    nk = amp
    sig = sig*const
    mu = mu*const
    a = mu/sig
    c = q**2*sig

    g = all_g1(t, nk, a, c)
    return g

by_Xs1 = jax.vmap(
    source3_1,
    in_axes=(None, None, 0, 0, 0)
)

source_matrix1 = jax.vmap(
    by_Xs1,
    in_axes=(None, 0, None, None, None)
)

def g1_matrix(q, t, amp, mu, sig):
    full = source_matrix1(t, q, amp, mu, sig)
    full = jnp.sum(full, axis=1)
    return full

# def g2_minus1_matrix(q, t, amp, mu, sig, beta):
#     g1 = g1_matrix(q, t, amp, mu, sig)
#     g2_minus1 = beta * g1**2
#     return g2_minus1

######### Dirac model ##########

def single_exp(q, t, D, amp):
    return amp * jnp.exp(-D * q**2 * t)

by_Xs = jax.vmap(
    single_exp,
    in_axes=(None, None, 0, 0)
)

by_t = jax.vmap(
    by_Xs,
    in_axes=(None, 0, None, None)
)

by_q = jax.vmap(
    by_t,
    in_axes=(0, None, None, None)
)

def g1_dirac(q, t, D, amp, const):
    D_ = D * const
    full = by_q(q, t, D_, amp)
    full = jnp.sum(full, axis=2)
    return full

# def g2_minus1_matrix_dirac(q, t, D, amp, beta, const):
#     amp = amp / jnp.sum(amp)
#     g1 = g1_dirac(q, t, D, amp, const)
#     g2_minus1 = beta * g1**2
#     return g2_minus1


# def gen_bounds_dirac(k):
#     return (
#         (
#             1e-9*jnp.ones(k),
#             1e-9*jnp.ones(k),
#             jnp.array([0.0])
#         ), (
#             1e-3*jnp.ones(k),
#             jnp.ones(k),
#             jnp.array([1.0])
#         )
#     )

def gen_bounds_dirac_g1(k):
    return (
        (
            1e-9*jnp.ones(k),
            1e-9*jnp.ones(k),
        ), (
            1e-3*jnp.ones(k),
            jnp.ones(k),
        )
    )



# %%
### Optimizers
min_k = 1
max_k = 3


# small_particle_bound = 1e-10 # 1 angstrom
# large_particle_bound = 1e-5 # 10 microns
# def gen_bounds_std(num_sources):
#     lower_bounds = (
#         1e-9*jnp.ones(num_sources),
#         jnp.array([1e-9])
#     )
#     upper_bounds = (
#         1e-3*jnp.ones(num_sources),
#         jnp.array([1.0])
#     )
#     return lower_bounds, upper_bounds

def gen_bounds_std_g1(num_sources):
    lower_bounds = (
        1e-9*jnp.ones(num_sources),
    )
    upper_bounds = (
        1e-3*jnp.ones(num_sources),
    )
    return lower_bounds, upper_bounds


def gen_bounds_normal_g1(num_sources):
    lower_bounds = (
        1e-9*jnp.ones(num_sources),
        1e-9*jnp.ones(num_sources),
        1e-9*jnp.ones(num_sources),
    )
    upper_bounds = (
        jnp.inf*jnp.ones(num_sources),
        1e-3*jnp.ones(num_sources),
        1e-3*jnp.ones(num_sources),
    )
    return lower_bounds, upper_bounds

# def extract_point(row):
#     # extract source amplitudes and positions to use as points for clustering
#     points = []
#     sol = row['sol']
#     D = sol[0]
#     amp = sol[1]
#     beta = sol[2]
#     if isinstance(amp, float):
#         D = jnp.array([D])
#         amp = jnp.array([amp])
#         beta = jnp.array([beta])
#     else:
#         D = jnp.array(D)
#         amp = jnp.array(amp)
#         beta = jnp.array(beta)
#     amp = amp/jnp.sum(amp)
#     for p in range(len(amp)):
#         point = jnp.stack([D[p], amp[p], beta[p]]).tolist()
#         points.append(point)
#     return points

def extract_point_g1(row):
    # extract source amplitudes and positions to use as points for clustering
    points = []
    sol = row['sol']
    D = sol[0]
    amp = sol[1]
    if isinstance(amp, float):
        D = jnp.array([D])
        amp = jnp.array([amp])
    else:
        D = jnp.array(D)
        amp = jnp.array(amp)
    amp = amp/jnp.sum(amp)
    for p in range(len(amp)):
        point = jnp.stack([D[p], amp[p]]).tolist()
        points.append(point)
    return points

# def clustering_preprocess(res):
#     res = res.copy().groupby('num_sources', group_keys=False)[res.columns.tolist()].apply(filter_quantile)
#     res['points'] = res.apply(extract_point, axis=1)
#     return res

def filter_quantile(res, col_to_filter='fval', quantile=0.25):
    mod_col = res[col_to_filter].apply(lambda x: jnp.inf if jnp.isnan(x) else x)
    return res[
        mod_col < mod_col.quantile(q=quantile)
    ]

def clustering_preprocess_g1(res):
    res = res.copy().groupby('num_sources', group_keys=False)[res.columns.tolist()].apply(filter_quantile)
    res['points'] = res.apply(extract_point_g1, axis=1)
    return res

# def process_res_dirac_(all_res, obs_size):
#     Forclusts = clustering_preprocess(all_res)
#     Forclusts = Forclusts.groupby('num_sources', group_keys=False)[Forclusts.columns.tolist()].apply(lambda group: result_analysis(
#         group['points'].sum(),
#         # group['normF'].mean(),
#         group['fval'].mean(),
#         obs_size,
#         group['num_sources'].iloc[0]
#     ))
#     Forclusts = Forclusts.set_index('num_source')

#     return Forclusts

# def process_res_dirac_(all_res, obs_size):
#     Forclusts = clustering_preprocess_g1(all_res)
#     Forclusts = Forclusts.groupby('num_sources', group_keys=False)[Forclusts.columns.tolist()].apply(lambda group: result_analysis(
#         group['points'].sum(),
#         # group['normF'].mean(),
#         group['fval'].mean(),
#         obs_size,
#         group['num_sources'].iloc[0]
#     ))
#     Forclusts = Forclusts.set_index('num_source')

#     return Forclusts


# def extract_point_std(row):
#     # extract source amplitudes and positions to use as points for clustering
#     points = []
#     sol = row['sol']
#     sig = sol[0]
#     beta = sol[1]
#     beta = jnp.array(beta)
#     if isinstance(sig, float):
#         sig = jnp.array([sig])
#     else:
#         sig = jnp.array(sig)
#     for p in range(len(sig)):
#         point = sig.reshape(-1, 1)[p].tolist()
#         points.append(point)
#     return points

# def clustering_preprocess_std(res):
#     res = res.copy().groupby('num_sources', group_keys=False)[res.columns.tolist()].apply(filter_quantile)
#     res['points'] = res.apply(extract_point_std, axis=1)
#     return res

# def process_res_std_(all_res, obs_size):
#     Forclusts = clustering_preprocess_std(all_res)
#     Forclusts = Forclusts.groupby('num_sources', group_keys=False)[Forclusts.columns.tolist()].apply(lambda group: result_analysis(
#         group['points'].sum(),
#         # group['normF'].mean(),
#         group['fval'].mean(),
#         obs_size,
#         group['num_sources'].iloc[0]
#     ))
#     Forclusts = Forclusts.set_index('num_source')

#     return Forclusts


def extract_point_std_g1(row):
    # extract source amplitudes and positions to use as points for clustering
    points = []
    sol = row['sol']
    sig = sol[0]
    if isinstance(sig, float):
        sig = jnp.array([sig])
    else:
        sig = jnp.array(sig)
    for p in range(len(sig)):
        point = sig.reshape(-1, 1)[p].tolist()
        points.append(point)
    return points

def clustering_preprocess_std_g1(res):
    res = res.copy().groupby('num_sources', group_keys=False)[res.columns.tolist()].apply(filter_quantile)
    res['points'] = res.apply(extract_point_std_g1, axis=1)
    return res

def process_res_std_g1_(all_res, obs_size):
    Forclusts = clustering_preprocess_std_g1(all_res)
    Forclusts = Forclusts.groupby('num_sources', group_keys=False)[Forclusts.columns.tolist()].apply(lambda group: result_analysis(
        group['points'].sum(),
        # group['normF'].mean(),
        group['fval'].mean(),
        obs_size,
        group['num_sources'].iloc[0]
    ))
    Forclusts = Forclusts.set_index('num_source')

    return Forclusts

# from dls_model import clustering_preprocess as clustering_preprocess_std_normal
def extract_point_std_normal_g1(row):
    # extract source amplitudes and positions to use as points for clustering
    points = []
    sol = row['sol']
    D = sol[0]
    amp = sol[1]
    if isinstance(amp, float):
        D = jnp.array([D])
        amp = jnp.array([amp])
    else:
        D = jnp.array(D)
        amp = jnp.array(amp)
    amp = amp/jnp.sum(amp)
    for p in range(len(amp)):
        point = jnp.stack([D[p], amp[p]]).tolist()
        points.append(point)
    return points

def clustering_preprocess_std_normal_g1(res):
    res = res.copy().groupby('num_sources', group_keys=False)[res.columns.tolist()].apply(filter_quantile)
    res['points'] = res.apply(extract_point_std_normal_g1, axis=1)
    return res

# def filter_quantile(res, col_to_filter='fval', quantile=0.25):
#     mod_col = res[col_to_filter].apply(lambda x: jnp.inf if jnp.isnan(x) else x)
#     return res[
#         mod_col < mod_col.quantile(q=quantile)
#     ]

def process_res_dirac_(all_res, obs_size):
    Forclusts = clustering_preprocess_g1(all_res)
    Forclusts = Forclusts.groupby('num_sources', group_keys=False)[Forclusts.columns.tolist()].apply(lambda group: result_analysis(
        group['points'].sum(),
        group['normF'].mean(),
        obs_size,
        group['num_sources'].iloc[0]
    ))
    Forclusts = Forclusts.set_index('num_source')

    return Forclusts


def process_res_normal_g1_(all_res, obs_size):
    Forclusts = clustering_preprocess_std_normal_g1(all_res)
    Forclusts = Forclusts.groupby('num_sources', group_keys=False)[Forclusts.columns.tolist()].apply(lambda group: result_analysis(
        group['points'].sum(),
        # group['normF'].mean(),
        group['fval'].mean(),
        obs_size,
        group['num_sources'].iloc[0]
    ))
    Forclusts = Forclusts.set_index('num_source')

    return Forclusts

def l_statistic(full_sols, clust_info, sill_threshold=0.6, p_threshold=0.05):
    p_values = {}
    errors = {}
    n_opt = 1
    for k in clust_info[clust_info['min_sillhouette_score'] > sill_threshold].index:
        current_errors = full_sols[full_sols['num_sources'] == k]['normF'].sort_values().to_list()
        errors[k] = current_errors

        # For the second valid k onwards, perform the statistical test
        if k > 1:
            # Get the errors from the previous valid k
            prev_errors = errors[k-1]
            
            # Wilcoxon rank-sum test to see if the new errors are significantly smaller
            # We use a one-sided test ('less') to check if the current error distribution
            # is stochastically less than the previous one.
            _, p_val = ranksums(current_errors, prev_errors, alternative='less')
            p_values[k] = p_val
            
            # If the result is significant, this k is a better model
            if p_val < p_threshold:
                n_opt = k
    return n_opt, p_values, errors

def l_statistic2(full_sols, clust_info, observations, q, t, sill_threshold=0.6, p_threshold=0.05):
    p_values = {}
    errors = {}
    n_opt = 1
    for k in clust_info[clust_info['min_sillhouette_score'] > sill_threshold].index:
        best_amp, best_D, best_sig = full_sols[full_sols['num_sources'] == k].sort_values('fval').iloc[0]['sol']
        # recon = g2_minus1_matrix(q, t, best_amp, best_D, best_sig, best_beta)
        recon = g1_matrix(q, t, best_amp, best_D, best_sig)
        current_errors = jnp.zeros(len(q))
        # current_errors = []
        for qs in range(len(q)):
            s_segment = observations[qs, :]
            shat_segment = recon[qs, :]
            
            # Calculate the relative vector norm (error)
            error_norm = jnp.linalg.norm(shat_segment - s_segment)
            signal_norm = jnp.linalg.norm(s_segment)
            # Avoid division by zero if a signal segment is all zeros
            current_errors = current_errors.at[qs].set(error_norm / signal_norm if signal_norm > 0 else 0)

        errors[k] = current_errors


        # For the second valid k onwards, perform the statistical test
        if k > 1:
            # Get the errors from the previous valid k
            prev_errors = errors[k-1]
            
            # Wilcoxon rank-sum test to see if the new errors are significantly smaller
            # We use a one-sided test ('less') to check if the current error distribution
            # is stochastically less than the previous one.
            _, p_val = ranksums(current_errors, prev_errors, alternative='less')
            p_values[k] = p_val
            
            # If the result is significant, this k is a better model
            if p_val < p_threshold:
                n_opt = k
    return n_opt, p_values, errors

# %% [markdown]
# ## run stuff

# %%
# import some experimental data
q, t, mix_1 = prep_data("exp_data/mix_1.csv")

process_res_dirac = functools.partial(process_res_dirac_, obs_size=mix_1.size)
# process_res_std = functools.partial(process_res_std_, obs_size=mix_1.size)
process_res_std_g1 = functools.partial(process_res_std_g1_, obs_size=mix_1.size)
process_res_normal = functools.partial(process_res_normal_g1_, obs_size=mix_1.size)

# %%
def generate_distributions_by_distance(amp_pairs, lower_means, mean_distances, std_devs):
    core_params = list(itertools.product(std_devs, lower_means, mean_distances))
    all_combinations = []
    for amp1, amp2 in amp_pairs:
        for std, mean1, dist in core_params:
            mean2 = mean1 + dist
            all_combinations.append([amp1, mean1, std, amp2, mean2, std])

    return jnp.array(all_combinations)
    # all_combinations_jnp = jnp.array(all_combinations)
    # _amp1, mean1, std1, _amp2, mean2, std2 = all_combinations_jnp.T
    # a1 = (mean1 / 10) <= std1
    # a2 = (mean1 / 2.5) >= std1
    # a3 = (mean2 / 10) <= std2
    # a4 = (mean2 / 2.5) >= std2
    # a5 = (mean1 != mean2)

    # valid_mask = jnp.all(
    #     jnp.array([
    #         a1, a2, a3, a4, a5
    #     ]),
    #     axis=0
    # )

    # return all_combinations_jnp[valid_mask]


amplitude_pairs = [[0.5, 0.5]]
means = jnp.arange(1.8e-6, 1.9e-5, 6e-6).tolist()
mean_diffs = jnp.arange(5e-7, 5.1e-6, 7.5e-7).tolist()
stds = jnp.arange(6.5e-7, 1.31e-6, 1.5e-7).tolist()

distanced_params = generate_distributions_by_distance(amplitude_pairs, means, mean_diffs, stds)
amp1, mean1, std1, amp2, mean2, std2 = distanced_params.T
valid_params = distanced_params = jnp.stack([amp1, amp2, mean1, mean2, std1, std2]).T.reshape(distanced_params.shape[0], 3, 2)


# %%
amp, mu, sig = valid_params[10]
# beta = 0.7

# %%
g1_matrix(q, t, amp, mu, sig)

# %%
[SCALING_CONST * std for std in stds]
# widths

# %%
[SCALING_CONST * mean for mean in means]
# lower means

# %%
[SCALING_CONST * mean_diff for mean_diff in mean_diffs]
# mean diff

# %%
# simulating shot (poisson) noise
def add_poisson_noise(g2_ideal, rand_key, average_counts_khz=1500, baseline=1.0):
    """
    g2_ideal - ideal g2 function
    average_counts_khz - average counts per channel in kHz
    seed - random seed for reproducibility
    """
    rand_key, subkey = jax.random.split(rand_key)
    mean_counts_per_channel = average_counts_khz * 100 # A proxy for total photon budget
    # The mean number of photons at each delay time τ is proportional to the ideal g2(τ)
    mean_photons_at_tau = mean_counts_per_channel * g2_ideal
    # Generate the noisy g2 data by drawing from a Poisson distribution
    # for each channel. This is the core of the shot noise simulation. 🎲
    noisy_counts = jax.random.poisson(subkey, mean_photons_at_tau)

    # Normalize the noisy counts to get the final noisy g2 function
    # The baseline of the noisy data is the average of the counts at long delay times
    noisy_baseline = jnp.mean(noisy_counts[:, -20:], axis=1) # Use last 20 channels for baseline
    g2_noisy = jax.vmap(lambda x, y: x/y)(noisy_counts, noisy_baseline)
    noisy_g2_minus_1 = g2_noisy - baseline  # Adjust the baseline to match the ideal g2

    return noisy_g2_minus_1, rand_key

def simulate_noisy_g2(
    g2_ideal: jnp.ndarray,
    rand_key: jax.random.PRNGKey,
    count_rate_khz: float,
    duration_s: float,
    baseline: float = 1.0,
    noise_scaling_factor: float = 0.1
) -> tuple[jnp.ndarray, jax.random.PRNGKey]:
    """
    Adds realistic Poisson noise to an ideal g2 autocorrelation function.

    This function simulates the shot noise inherent in a DLS experiment based on
    the instrument's count rate and the total measurement time.

    Args:
        g2_ideal: The ideal, noiseless g2 function (should not have the baseline subtracted).
                  Shape should be (batch, num_channels).
        rand_key: JAX random key for reproducibility.
        count_rate_khz: The average photon count rate in kHz (e.g., 20-45 from the manual).
        duration_s: The total duration of the experiment in seconds (e.g., 10, 30, 60).
        baseline: The theoretical baseline of the correlation function (typically 1.0).
        noise_scaling_factor: An empirical factor to match simulation to a real
                              correlator's output. It bridges the gap between total
                              photons and the statistical quality of the g2 function.
                              A value between 0.05 and 0.2 is a good starting point.

    Returns:
        A tuple containing:
        - noisy_g2_minus_1: The noisy g2 function with the baseline subtracted.
        - rand_key: The updated JAX random key.
    """
    # 1. Calculate the effective number of photon counts that contribute to the
    #    baseline of the correlation function. This is our "photon budget" and
    #    is the primary determinant of the noise level. It combines the
    #    instantaneous rate with the total measurement time.
    #    Total photons = count_rate_khz * 1000 * duration_s.
    #    The noise_scaling_factor adjusts this to better match the statistics
    #    of a real hardware correlator's averaging process.
    mean_counts_at_baseline = (
        count_rate_khz * 1000 * duration_s * noise_scaling_factor
    )

    # 2. The mean number of photons at each delay time τ is proportional to the ideal g2(τ).
    #    This creates the shape of the correlation function.
    mean_photons_at_tau = mean_counts_at_baseline * g2_ideal

    # 3. Generate the noisy data by drawing from a Poisson distribution for each channel.
    #    This is the core of the shot noise simulation.
    rand_key, subkey = jax.random.split(rand_key)
    noisy_counts = jax.random.poisson(subkey, mean_photons_at_tau)

    # 4. Normalize the noisy counts to get the final noisy g2 function.
    #    A robust method for finding the baseline of the noisy data is to average
    #    the counts from the last ~10% of the channels, where the function has decayed.
    num_channels_for_baseline = noisy_counts.shape[-1] // 10
    noisy_baseline = jnp.mean(noisy_counts[..., -num_channels_for_baseline:], axis=-1, keepdims=True)

    # Avoid division by zero if the baseline is somehow zero
    noisy_baseline = jnp.where(noisy_baseline == 0, 1.0, noisy_baseline)

    g2_noisy = noisy_counts / noisy_baseline

    # 5. Adjust by the theoretical baseline to center the result around 0.
    noisy_g2_minus_1 = g2_noisy - baseline

    return noisy_g2_minus_1, rand_key


# def gen_data(
#         q,
#         t,
#         valid_params,
#         ensemble_size,
#         gen_beta=0.7,
#         baseline=1.0,
#         rand_key=jax.random.key(1337),
#         count_rate_khz=45.0,
#         experiment_measurement_time_s=30.0,
#         noise_scaling_factor=0.1
#     ):
#     clean_obs_list = []
#     noisy_obs_list = []
#     noise_errors = []
#     snrs = []
#     for i in range(valid_params.shape[0]):
#         amp, mu, sig = valid_params[i]
#         clean_g1 = g1_matrix(q, t, amp, mu, sig)
#         g2_ideal = baseline + gen_beta*(clean_g1**2)


#         # ensemble_observations = []
#         # for _ in range(ensemble_size):
#         #     noisy_g2_minus_1, rand_key = add_poisson_noise(g2_ideal, rand_key, average_counts_khz=average_counts_khz, baseline=baseline)
#         #     ensemble_observations.append(noisy_g2_minus_1)
#         # noisy_g2_minus_1 = jnp.mean(jnp.array(ensemble_observations), axis=0)

#         # noisy_g2_minus_1, rand_key = add_poisson_noise(g2_ideal, rand_key, average_counts_khz=average_counts_khz, baseline=baseline)

#         ensemble_observations = []
#         for _ in range(ensemble_size):
#             noisy_g2_minus_1, rand_key = simulate_noisy_g2(
#                 g2_ideal,
#                 rand_key,
#                 count_rate_khz=count_rate_khz,
#                 duration_s=experiment_measurement_time_s,
#                 baseline=baseline,
#                 noise_scaling_factor=noise_scaling_factor
#             )
#             ensemble_observations.append(noisy_g2_minus_1)
#         noisy_g2_minus_1 = jnp.mean(jnp.array(ensemble_observations), axis=0)

#         g2_ideal_minus1 = g2_ideal - baseline
#         clean_obs_list.append(g2_ideal_minus1)

#         snr = gen_beta / jnp.std((noisy_g2_minus_1 - g2_ideal_minus1))
#         snrs.append(snr)

#         r = g2_ideal_minus1 - noisy_g2_minus_1
#         rmse = jnp.sqrt(jnp.sum(jnp.square(r)) / g2_ideal_minus1.size)
#         noise_errors.append(rmse)

#         noisy_obs_list.append(noisy_g2_minus_1)
#         # noisy_obs_list.append(g2_ideal - baseline)

#     return clean_obs_list, noisy_obs_list, noise_errors, snrs


# # average_counts_khz = 1500
# gen_beta = 0.7

# ensemble_size = 1

# clean_obs_list, noisy_obs_list, noise_errors, snrs = gen_data(
#     q,
#     t,
#     valid_params,
#     ensemble_size,
#     gen_beta=gen_beta,
#     baseline=1.0,
#     rand_key=jax.random.key(1337),
#     count_rate_khz=20.0, # average count rate for HeNe laser
#     experiment_measurement_time_s=30.0, # total measurement time in seconds
#     noise_scaling_factor=0.1
# )

clean_obs_list = []
for i in range(valid_params.shape[0]):
    amp, mu, sig = valid_params[i]
    clean_g1 = g1_matrix(q, t, amp, mu, sig)
    clean_obs_list.append(clean_g1)


# print(f"error due to noise (avg rmse): {jnp.average(jnp.array(noise_errors))}")
# print(f"average SNR: {jnp.average(jnp.array(snrs))}")



# %%
# set up the multi-phase optimization

optimizer_dirac_single = NewHNMFOptimizer(
# optimizer_dirac_single = PerturbanceHNMFOptimizer(
    # model_fn=g2_minus1_matrix_dirac,
    # param_generator=InitParamsGenerator2(gen_bounds_dirac),
    # bound_generator=gen_bounds_dirac,
    model_fn=g1_dirac,
    param_generator=InitParamsGenerator2(gen_bounds_dirac_g1),
    bound_generator=gen_bounds_dirac_g1,
    input_args = ('q', 't'),
    param_args=('D', 'amp'),
    constants = {"const": SCALING_CONST},
    min_k=min_k,
    max_k=max_k,
    nsim=100
)


def gen_bounds_normal_std(num_sources):
    lower_bounds = (
        # 1e-9*jnp.ones(num_sources),
        # 1e-9*jnp.ones(num_sources),
        1e-9*jnp.ones(num_sources),
        # jnp.array([0.0])
    )
    upper_bounds = (
        # jnp.inf*jnp.ones(num_sources),
        # 1e-3*jnp.ones(num_sources),
        1e-3*jnp.ones(num_sources),
        # jnp.array([1.0])
    )
    return lower_bounds, upper_bounds

std_opts = {}
for k in range(min_k, max_k + 1):
    std_opt = NewHNMFOptimizer(
        model_fn=g1_matrix,
        param_generator=InitParamsGenerator2(gen_bounds_normal_std),
        bound_generator=gen_bounds_normal_std,
        input_args = ('q', 't', 'amp', 'mu'),
        param_args=('sig',),
        constants = {},
        min_k=k,
        max_k=k,
        nsim=20
    )
    std_opts[k] = (std_opt)


def gen_bounds_normal_final(num_sources):
    lower_bounds = (
        1e-9*jnp.ones(num_sources),
        1e-9*jnp.ones(num_sources),
        1e-9*jnp.ones(num_sources),
        # jnp.array([0.0])
    )
    upper_bounds = (
        jnp.inf*jnp.ones(num_sources),
        1e-3*jnp.ones(num_sources),
        1e-3*jnp.ones(num_sources),
        # jnp.array([1.0])
    )
    return lower_bounds, upper_bounds

final_opt = NewHNMFOptimizer(
# final_opt = PerturbanceHNMFOptimizer(
    # model_fn=g1_matrix,
    model_fn=g1_matrix,
    param_generator=InitParamsGenerator2(gen_bounds_normal_final),
    bound_generator=gen_bounds_normal_final,
    input_args = ('q', 't'),
    param_args=('amp', 'mu', 'sig'),
    constants = {},
    min_k=min_k,
    max_k=max_k,
    nsim=100
)


inputs = (q, t)
opt_options = {
    'fatol': 1e-14,
    'frtol': 0,
    'maxiter': 2000,
    'gatol': 1e-12
}



# %%

# %%
# rand_key = jax.random.key(495)

final_full_sols = []
final_clust_sols = []

# quad_final_full_sols = []
# quad_final_clust_sols = []

# %%
class InitParamFeeder:
    def __init__(self, params_list_map):
        self.params_list_map = params_list_map
        self.counter = {k: 0 for k in params_list_map.keys()}

    def __call__(self, num_sources):
        params = self.params_list_map[num_sources][self.counter[num_sources]]
        self.counter[num_sources] += 1
        return params

















# for i in range(len(final_clust_sols), len(noisy_obs_list)):
def run_opt(i):
    rand_key = jax.random.key(495 + i)
    # rand_key = jax.random.key(495)
    for _j in range(i):
        rand_key, subkey1, subkey2, subkey3 = jax.random.split(rand_key, 4)
    # observations = noisy_obs_list[i]
    observations = clean_obs_list[i]


    # phase 1 - fit dirac model, gives centers and amplitudes
    g2_res = optimizer_dirac_single((q, t), observations, opt_options=opt_options)
    clust_sol = process_res_dirac(g2_res)

    print("finished phase 1 for pair", i)


    # dirac_sols.append((D, amp, beta))

    # phase 2 - convert dirac to normal, only fit the standard deviation of the gaussians
    # centers and amplitudes are taken from the first phase and fixed

    params_to_feed = {}
    for k in clust_sol.index:
        D, amp = clust_sol.loc[k]['centers']
        if not isinstance(D, jnp.ndarray):
            D = jnp.array(D, ndmin=1)
        if not isinstance(amp, jnp.ndarray):
            amp = jnp.array(amp, ndmin=1)
        std_sols = std_opts[k]((q, t, amp, D), observations, opt_options=opt_options)
        sig_sol = process_res_std_g1(std_sols)['centers'].iloc[0][0]
        if not isinstance(sig_sol, jnp.ndarray):
            sig_sol = jnp.array(sig_sol, ndmin=1)

        nsim = 100
        params_to_feed[k] = []
        for j in range(nsim):
            rand_key, subkey1, subkey2, subkey3 = jax.random.split(rand_key, 4)
            D_ = D + jax.random.normal(subkey1, D.shape) * 0.05 * D
            sig_ = sig_sol + jax.random.normal(subkey2, sig_sol.shape) * 0.05 * sig_sol
            amp_ = amp + jax.random.normal(subkey3, amp.shape) * 0.05 * amp
            params_to_feed[k].append((amp_, D_, sig_))

    print("finished phase 2 for pair", i)

    # final phase - fit with all parameters free

    final_opt.reset_param_generator(InitParamFeeder(params_to_feed))

    final_g2_sols = final_opt((q, t), observations, opt_options=opt_options)
    final_full_sols.append(final_g2_sols)
    final_clust_sol = process_res_normal(final_g2_sols)
    final_clust_sols.append(final_clust_sol)

    output_dir = 'results'
    Path(output_dir).mkdir(exist_ok=True)
    result = {
        'index': i,
        # 'final_full_sol': final_g2_sols,
        'final_clust_sol': final_clust_sol
    }

    with open(f'{output_dir}/result_{i:04d}.pkl', 'wb') as f:
        pickle.dump(result, f)
    
    print(f"[Job {i}] Done")


    # with open('distance_clust_sols.pkl', 'wb') as f:
    #     pickle.dump(final_clust_sols, f)

    # print(f"(noisy) Pair {i} done\n\n")

import sys

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python this_script.py <index>")
        sys.exit(1)
    
    index = int(sys.argv[1])
    run_opt(index)











# %%
### save solutions with pickle ###

# with open('distance_sols.pkl', 'wb') as f:
#     ress = (final_full_sols, final_clust_sols)
#     pickle.dump(ress, f)


# save final_full_sols with pickle
# with open('quad_final_full_sols.pkl', 'wb') as f:
#     pickle.dump(quad_final_full_sols, f)


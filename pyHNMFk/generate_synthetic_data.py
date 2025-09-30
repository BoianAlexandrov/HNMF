import functools
import itertools

import jax.numpy as jnp
import jax
jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_compilation_cache_max_size", 2**28 - 1)


jax.config.update("jax_explain_cache_misses", True)
import pandas as pd



from dls_model import diffusion_coef, scatter_vector, g1_matrix, process_res_dirac_, process_res_std_, process_res_std_g1_, process_res_normal_, SCALING_CONST
# plt.style.use('Solarize_Light2')

# from IPython.display import Markdown, display



# %% [markdown]
# ## models, optimizers, helpers, etc

# %%
# Plotting theme setup
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

# import some experimental data
q, t, observations_100 = prep_data("exp_data/stock_100nm.csv")
q, t, observations_200 = prep_data("exp_data/stock_200nm.csv")
q, t, observations_500 = prep_data("exp_data/stock_500nm.csv")
q, t, observations_1000 = prep_data("exp_data/stock_1000nm.csv")
q, t, mix_1 = prep_data("exp_data/mix_1.csv")
q, t, mix_2 = prep_data("exp_data/mix_2.csv")
q, t, mix_3 = prep_data("exp_data/mix_3.csv")
q, t, mix_4 = prep_data("exp_data/mix_4.csv")

q, t, observations_100_g2 = prep_data_g2("exp_data/stock_100nm.csv")
q, t, observations_200_g2 = prep_data_g2("exp_data/stock_200nm.csv")
q, t, observations_500_g2 = prep_data_g2("exp_data/stock_500nm.csv")
q, t, observations_1000_g2 = prep_data_g2("exp_data/stock_1000nm.csv")
q, t, mix_2_g2 = prep_data_g2("exp_data/mix_2.csv")
q, t, mix_1_g2 = prep_data_g2("exp_data/mix_1.csv")
q, t, mix_3_g2 = prep_data_g2("exp_data/mix_3.csv")
q, t, mix_4_g2 = prep_data_g2("exp_data/mix_4.csv")


mix_avg_g2 = jnp.mean(jnp.stack([mix_1_g2, mix_2_g2, mix_3_g2, mix_4_g2]), axis=0)


process_res_dirac = functools.partial(process_res_dirac_, obs_size=mix_1.size)
process_res_std = functools.partial(process_res_std_, obs_size=mix_1.size)
process_res_std_g1 = functools.partial(process_res_std_g1_, obs_size=mix_1.size)
process_res_normal = functools.partial(process_res_normal_, obs_size=mix_1.size)

# %%
def generate_and_filter_distributions(amp_pairs, mean_params, std_params):
    # Generate combinations for the mean and standard deviation of both distributions
    mean_std_combinations = list(itertools.product(mean_params, std_params, mean_params, std_params))

    # Combine the user-provided amplitude pairs with the mean/std combinations
    all_combinations = []
    for amp1, amp2 in amp_pairs:
        for mean1, std1, mean2, std2 in mean_std_combinations:
            all_combinations.append([amp1, mean1, std1, amp2, mean2, std2])
    
    # Convert the list of combinations to a JAX array for efficient processing
    if not all_combinations:
        return jnp.array([]) # Return empty array if no combinations were generated
        
    all_combinations_jnp = jnp.array(all_combinations)
    _amp1, mean1, std1, _amp2, mean2, std2 = all_combinations_jnp.T
    a1 = (mean1 / 10) <= std1
    a2 = (mean1 / 2.5) >= std1
    a3 = (mean2 / 10) <= std2
    a4 = (mean2 / 2.5) >= std2
    a5 = (mean1 != mean2)

    valid_mask = jnp.all(
        jnp.array([
            a1, a2, a3, a4, a5
        ]),
        axis=0
    )

    # valid_mask = jnp.logical_and(jnp.logical_and(a1, a2), jnp.logical_and(a3, a4))

    return all_combinations_jnp[valid_mask]

radii = jnp.arange(50, 501, 50.) * 1e-9
_diff_coefs = diffusion_coef(radii)
scaled_diff_coefs = _diff_coefs / SCALING_CONST

amplitude_pairs = [[0.2, 0.8], [0.5, 0.5], [0.3, 0.7]]
mean_params = scaled_diff_coefs.tolist()
std_dev_params = jnp.linspace(5e-8, 1e-5, 6).tolist()

# Generate and filter the distributions
valid_params = generate_and_filter_distributions(
    amplitude_pairs, 
    mean_params, 
    std_dev_params
)

amp1, mean1, std1, amp2, mean2, std2 = valid_params.T
valid_params = jnp.stack([amp1, amp2, mean1, mean2, std1, std2]).T.reshape(valid_params.shape[0], 3, 2)

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


def gen_data(
        q,
        t,
        valid_params,
        ensemble_size,
        gen_beta=0.7,
        baseline=1.0,
        rand_key=jax.random.key(1337),
        count_rate_khz=45.0,
        experiment_measurement_time_s=30.0,
        noise_scaling_factor=0.1
    ):
    clean_obs_list = []
    noisy_obs_list = []
    noise_errors = []
    snrs = []
    for i in range(valid_params.shape[0]):
        amp, mu, sig = valid_params[i]
        clean_g1 = g1_matrix(q, t, amp, mu, sig)
        g2_ideal = baseline + gen_beta*(clean_g1**2)


        # ensemble_observations = []
        # for _ in range(ensemble_size):
        #     noisy_g2_minus_1, rand_key = add_poisson_noise(g2_ideal, rand_key, average_counts_khz=average_counts_khz, baseline=baseline)
        #     ensemble_observations.append(noisy_g2_minus_1)
        # noisy_g2_minus_1 = jnp.mean(jnp.array(ensemble_observations), axis=0)

        # noisy_g2_minus_1, rand_key = add_poisson_noise(g2_ideal, rand_key, average_counts_khz=average_counts_khz, baseline=baseline)

        ensemble_observations = []
        for _ in range(ensemble_size):
            noisy_g2_minus_1, rand_key = simulate_noisy_g2(
                g2_ideal,
                rand_key,
                count_rate_khz=count_rate_khz,
                duration_s=experiment_measurement_time_s,
                baseline=baseline,
                noise_scaling_factor=noise_scaling_factor
            )
            ensemble_observations.append(noisy_g2_minus_1)
        noisy_g2_minus_1 = jnp.mean(jnp.array(ensemble_observations), axis=0)

        g2_ideal_minus1 = g2_ideal - baseline
        clean_obs_list.append(g2_ideal_minus1)

        snr = gen_beta / jnp.std((noisy_g2_minus_1 - g2_ideal_minus1))
        snrs.append(snr)

        r = g2_ideal_minus1 - noisy_g2_minus_1
        rmse = jnp.sqrt(jnp.sum(jnp.square(r)) / g2_ideal_minus1.size)
        noise_errors.append(rmse)

        noisy_obs_list.append(noisy_g2_minus_1)
        # noisy_obs_list.append(g2_ideal - baseline)

    return clean_obs_list, noisy_obs_list, noise_errors, snrs


# average_counts_khz = 1500
gen_beta = 0.7

ensemble_size = 30

clean_obs_list, noisy_obs_list, noise_errors, snrs = gen_data(
    q,
    t,
    valid_params,
    ensemble_size,
    gen_beta=gen_beta,
    baseline=1.0,
    rand_key=jax.random.key(1337),
    count_rate_khz=20.0, # average count rate for HeNe laser
    experiment_measurement_time_s=30.0, # total measurement time in seconds
    noise_scaling_factor=0.1
)

print(f"error due to noise (avg rmse): {jnp.average(jnp.array(noise_errors))}")
print(f"average SNR: {jnp.average(jnp.array(snrs))}")

# save to numpy compressed file
import numpy as np
np.savez_compressed(
    "synthetic_data.npz",
    clean_obs=np.array(clean_obs_list),
    noisy_obs=np.array(noisy_obs_list),
    params=np.array(valid_params),
    noise_errors=np.array(noise_errors),
    snrs=np.array(snrs)
)

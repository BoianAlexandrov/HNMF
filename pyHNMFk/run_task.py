import base64
import io
import pickle
import functools
import itertools

import jax.numpy as jnp
import jax
jax.config.update("jax_platforms", "cpu")
# set cache size to 1GB
jax.config.update("jax_compilation_cache_max_size", 2**30 - 1)

# jax.config.update("jax_log_compiles", True)
# jax.config.update("jax_compiler_detailed_logging_min_ops", 50)

jax.config.update("jax_explain_cache_misses", True)

import pandas as pd
from scipy.io import loadmat, savemat
from scipy.stats import ranksums

from hnmf_tr_optimizer.hnmf_optimizer import HNMFOptimizer, NewHNMFOptimizer, PerturbanceHNMFOptimizer, RedoHNMFOptimizer
from hnmf_tr_optimizer.clusts import result_analysis

from dls_model import diffusion_coef, scatter_vector, g1_matrix, process_res_dirac_, process_res_std_, process_res_std_g1_, process_res_normal_, SCALING_CONST, g2_minus1_matrix_dirac, g2_minus1_matrix, g2_minus1_quadrature, normal_distribution, normal_distributions, InitParamsGenerator2, gen_bounds_dirac

min_k = 1
max_k = 3


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
        best_amp, best_D, best_sig, best_beta = full_sols[full_sols['num_sources'] == k].sort_values('fval').iloc[0]['sol']
        recon = g2_minus1_matrix(q, t, best_amp, best_D, best_sig, best_beta)
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

# %%
# import some experimental data
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


q, t, dummy_data = prep_data("exp_data/stock_100nm.csv")


process_res_dirac = functools.partial(process_res_dirac_, obs_size=dummy_data.size)
process_res_std = functools.partial(process_res_std_, obs_size=dummy_data.size)
process_res_std_g1 = functools.partial(process_res_std_g1_, obs_size=dummy_data.size)
process_res_normal = functools.partial(process_res_normal_, obs_size=dummy_data.size)

import numpy as np
data = np.load("synthetic_data.npz", allow_pickle=True)
clean_obs_list = jnp.array(data['clean_obs'])
noisy_obs_list = jnp.array(data['noisy_obs'])
valid_params = jnp.array(data['params'])
noise_errors = jnp.array(data['noise_errors'])
snrs = jnp.array(data['snrs'])
gen_beta = 0.7


# def rilt_observation_matrix(q, t, possible_D, x):
#     A = jnp.exp(jnp.einsum('i,j,k->ijk', -possible_D, q**2, t))
#     return jnp.einsum('i,ijk->jk', x, A)

# def zero_at_ends_rilt(q, t, possible_D, x):
#     x_ = x.at[0].set(0.0).at[-1].set(0.0)
#     return rilt_observation_matrix(q, t, possible_D, x_)

# def dummy_bounds(k):
#     return (jnp.zeros(k),), (1e1 * jnp.ones(k),)


# def L1_norm(x, alpha):
#     return alpha * jnp.sum(jnp.abs(x))

# L1_grad = jax.grad(L1_norm)
# L1_hess = jax.hessian(L1_norm)

# def L1_regularizer(x, alpha=1.0):
#     return L1_norm(x, alpha), L1_grad(x, alpha), L1_hess(x, alpha)


# def L2_norm(x, alpha):
#     return alpha * jnp.sqrt(jnp.sum(jnp.square(x)))

# L2_grad = jax.grad(L2_norm)
# L2_hess = jax.hessian(L2_norm)

# def L2_regularizer(x, alpha=1.0):
#     return L2_norm(x, alpha), L2_grad(x, alpha), L2_hess(x, alpha)

# # possible_D = jnp.logspace(-6.5, -4.5, 30) * SCALING_CONST
# possible_D = jnp.linspace(1e-7, 3e-5, 40) * SCALING_CONST

# # contin_opt = HNMFOptimizer(
# contin_opt = NewHNMFOptimizer(
# # contin_opt = PerturbanceHNMFOptimizer(
# # contin_opt = RedoHNMFOptimizer(
#     model_fn=zero_at_ends_rilt,
#     param_generator=InitParamsGenerator2(dummy_bounds),
#     bound_generator=dummy_bounds,
#     input_args = ('q', 't', 'possible_D'),
#     param_args=('x'),
#     constants = {},
#     min_k=len(possible_D),
#     max_k=len(possible_D),
#     nsim=5,
#     regularizer_fn=functools.partial(L2_regularizer, alpha=0.005)
#     # regularizer_fn=functools.partial(L1_regularizer, alpha=1.0)
# )



# # %%
# rilt_sols_list = []


# # %%
# rand_key = jax.random.key(808)
# for i in range(len(rilt_sols_list), len(noisy_obs_list)):
#     # if i > 3:
#     #     break
#     all_res = contin_opt((q, t, possible_D), noisy_obs_list[i], opt_options={
#         'fatol': 1e-14,
#         'frtol': 0,
#         'maxiter': 2000,
#         'gatol': 1e-12
#     })
#     sols = jnp.stack(all_res.sort_values('fval')['sol'].apply(lambda l: l[0]).tolist())
#     sols = sols.at[:, 0].set(0).at[:, -1].set(0)
#     rilt_sols_list.append(sols)
#     print(f"Pair {i+1} done\n\n")



# # og_res = rilt_sols_list[0]
# new_res = rilt_sols_list[0]


# # %%
# rand_key = jax.random.key(811)
# for i in range(len(rilt_sols_list), len(noisy_obs_list)):
#     if i > 3:
#         break
#     perturbed_obs_list = []
#     for j in range(contin_opt.nsim):
#         rand_key, subkey = jax.random.split(rand_key)
#         perturbed_obs = jax.random.normal(subkey, shape=noisy_obs_list[i].shape) * gen_beta * 0.00000001 + noisy_obs_list[i]
#         perturbed_obs_list.append(perturbed_obs)

#     all_res = contin_opt((q, t, possible_D), perturbed_obs_list, opt_options={
#         'fatol': 1e-14,
#         'frtol': 0,
#         'maxiter': 2000,
#         'gatol': 1e-12
#     })
#     sols = jnp.stack(all_res.sort_values('fval')['sol'].apply(lambda l: l[0]).tolist())
#     sols = sols.at[:, 0].set(0).at[:, -1].set(0)
#     rilt_sols_list.append(sols)
#     print(f"Pair {i+1} done\n\n")
# perturbed_res = rilt_sols_list[0]


# %%
# set up the multi-phase optimization

# optimizer_dirac_single = HNMFOptimizer(
# optimizer_dirac_single = PerturbanceHNMFOptimizer(
optimizer_dirac_single = NewHNMFOptimizer(
    model_fn=g2_minus1_matrix_dirac,
    param_generator=InitParamsGenerator2(gen_bounds_dirac),
    bound_generator=gen_bounds_dirac,
    input_args = ('q', 't'),
    param_args=('D', 'amp', 'beta'),
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
        model_fn=g2_minus1_matrix,
        param_generator=InitParamsGenerator2(gen_bounds_normal_std),
        bound_generator=gen_bounds_normal_std,
        input_args = ('q', 't', 'amp', 'mu', 'beta'),
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
        jnp.array([0.0])
    )
    upper_bounds = (
        jnp.inf*jnp.ones(num_sources),
        1e-3*jnp.ones(num_sources),
        1e-3*jnp.ones(num_sources),
        jnp.array([1.0])
    )
    return lower_bounds, upper_bounds

# final_opt = HNMFOptimizer(
# final_opt = PerturbanceHNMFOptimizer(
final_opt = NewHNMFOptimizer(
    # model_fn=g1_matrix,
    model_fn=g2_minus1_matrix,
    param_generator=InitParamsGenerator2(gen_bounds_normal_final),
    bound_generator=gen_bounds_normal_final,
    input_args = ('q', 't'),
    param_args=('amp', 'mu', 'sig', 'beta'),
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


# final_full_sols = []
# final_clust_sols = []


class InitParamFeeder:
    def __init__(self, params_list_map):
        self.params_list_map = params_list_map
        self.counter = {k: 0 for k in params_list_map.keys()}

    def __call__(self, num_sources):
        params = self.params_list_map[num_sources][self.counter[num_sources]]
        self.counter[num_sources] += 1
        return params

# for i in range(len(final_clust_sols), len(noisy_obs_list)):
#     # observations = noisy_obs_list[i]
#     observations = clean_obs_list[i]


#     # phase 1 - fit dirac model, gives centers and amplitudes
#     g2_res = optimizer_dirac_single((q, t), observations, opt_options=opt_options)
#     clust_sol = process_res_dirac(g2_res)

#     print("finished phase 1 for pair", i)


#     # dirac_sols.append((D, amp, beta))

#     # phase 2 - convert dirac to normal, only fit the standard deviation of the gaussians
#     # centers and amplitudes are taken from the first phase and fixed

#     params_to_feed = {}
#     for k in clust_sol.index:
#         D, amp, betas = clust_sol.loc[k]['centers']
#         if not isinstance(D, jnp.ndarray):
#             D = jnp.array(D, ndmin=1)
#         if not isinstance(amp, jnp.ndarray):
#             amp = jnp.array(amp, ndmin=1)
#         if not isinstance(betas, jnp.ndarray):
#             betas = jnp.array(betas, ndmin=1)
#         beta = betas[0:1]
#         std_sols = std_opts[k]((q, t, amp, D, beta), observations, opt_options=opt_options)
#         sig_sol = process_res_std_g1(std_sols)['centers'].iloc[0][0]
#         if not isinstance(sig_sol, jnp.ndarray):
#             sig_sol = jnp.array(sig_sol, ndmin=1)

#         nsim = 100
#         params_to_feed[k] = []
#         for j in range(nsim):
#             rand_key, subkey1, subkey2, subkey3 = jax.random.split(rand_key, 4)
#             D_ = D + jax.random.normal(subkey1, D.shape) * 0.05 * D
#             sig_ = sig_sol + jax.random.normal(subkey2, sig_sol.shape) * 0.05 * sig_sol
#             amp_ = amp + jax.random.normal(subkey3, amp.shape) * 0.05 * amp
#             params_to_feed[k].append((amp_, D_, sig_, beta))

#     print("finished phase 2 for pair", i)

#     # final phase - fit with all parameters free

#     final_opt.reset_param_generator(InitParamFeeder(params_to_feed))

#     final_g2_sols = final_opt((q, t), observations, opt_options=opt_options)
#     final_full_sols.append(final_g2_sols)
#     final_clust_sol = process_res_normal(final_g2_sols)
#     final_clust_sols.append(final_clust_sol)

#     with open('distance_clust_sols.pkl', 'wb') as f:
#         pickle.dump(final_clust_sols, f)

#     print(f"(no-noise) Pair {i} done\n\n")
clean_obs_list = jnp.array(data['clean_obs'])
noisy_obs_list = jnp.array(data['noisy_obs'])
valid_params = jnp.array(data['params'])
noise_errors = jnp.array(data['noise_errors'])
snrs = jnp.array(data['snrs'])
gen_beta = 0.7

import sys
from pathlib import Path

def process_observation(index, output_dir="results"):
    """Process a single observation at the given index"""
    
    # Load shared data (you'll need to save this once before running parallel jobs)
    # Initialize random key with index for reproducibility
    rand_key = jax.random.PRNGKey(42 + index)
    
    i = index
    observations = clean_obs_list[i]
    
    # Phase 1 - fit dirac model
    g2_res = optimizer_dirac_single((q, t), observations, opt_options=opt_options)
    clust_sol = process_res_dirac(g2_res)
    
    print(f"[Job {i}] Finished phase 1")
    
    # Phase 2 - convert dirac to normal
    params_to_feed = {}
    for k in clust_sol.index:
        D, amp, betas = clust_sol.loc[k]['centers']
        if not isinstance(D, jnp.ndarray):
            D = jnp.array(D, ndmin=1)
        if not isinstance(amp, jnp.ndarray):
            amp = jnp.array(amp, ndmin=1)
        if not isinstance(betas, jnp.ndarray):
            betas = jnp.array(betas, ndmin=1)
        beta = betas[0:1]
        
        std_sols = std_opts[k]((q, t, amp, D, beta), observations, opt_options=opt_options)
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
            params_to_feed[k].append((amp_, D_, sig_, beta))
    
    print(f"[Job {i}] Finished phase 2")
    
    # Final phase - fit with all parameters free
    final_opt.reset_param_generator(InitParamFeeder(params_to_feed))
    final_g2_sols = final_opt((q, t), observations, opt_options=opt_options)
    final_clust_sol = process_res_normal(final_g2_sols)
    
    # Save individual results
    Path(output_dir).mkdir(exist_ok=True)
    result = {
        'index': i,
        'final_full_sol': final_g2_sols,
        'final_clust_sol': final_clust_sol
    }
    
    with open(f'{output_dir}/result_{i:04d}.pkl', 'wb') as f:
        pickle.dump(result, f)
    
    print(f"[Job {i}] Done")
    return i

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python process_single_observation.py <index>")
        sys.exit(1)
    
    index = int(sys.argv[1])
    process_observation(index)


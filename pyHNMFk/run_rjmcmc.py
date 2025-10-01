import pickle
import sys
from pathlib import Path
import jax
jax.config.update("jax_enable_x64", True)  # Enable 64-bit precision
# jax.config.update("jax_debug_nans", True)
jax.config.update("jax_platforms", "cpu")
import jax.numpy as jnp
import jax.random as random


import matplotlib.pyplot as plt
import functools
import pandas as pd

from dls_model import diffusion_coef, scatter_vector, g2_minus1_matrix, g1_matrix, process_res_dirac_, process_res_std_, process_res_std_g1_, process_res_normal_, SCALING_CONST

from rjmcmc import rjmcmc_sampler_dls_bd, rjmcmc_sampler_dls_paper, DLSState, prep_data, prep_data_g2

def plot_dls_results(samples, data, q_values, t_values, output_dir, index, burn_in=1000, true_params=None):
    """Plot comprehensive results from DLS RJMCMC."""
    plt.clf()
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    # Remove burn-in
    k_samples = samples['k'][burn_in:]
    beta_samples = samples['beta'][burn_in:]
    sigma_samples = samples['sigma_noise'][burn_in:]
    logpost_samples = samples['log_posterior'][burn_in:]
    
    # 1. Trace plot of k
    axes[0, 0].plot(k_samples)
    if true_params and 'k' in true_params:
        axes[0, 0].axhline(true_params['k'], color='red', linestyle='--', label=f'True k={true_params["k"]}')
    axes[0, 0].set_title("Trace: Number of Components (k)")
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("k")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Posterior of k
    unique_k, counts = jnp.unique(k_samples, return_counts=True)
    posterior_probs = counts / len(k_samples)
    axes[0, 1].bar(unique_k, posterior_probs, alpha=0.7)
    if true_params and 'k' in true_params:
        axes[0, 1].axvline(true_params['k'], color='red', linestyle='--')
    axes[0, 1].set_title("Posterior Distribution of k")
    axes[0, 1].set_xlabel("Number of Components")
    axes[0, 1].set_ylabel("Probability")
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Log posterior trace
    axes[0, 2].plot(logpost_samples)
    axes[0, 2].set_title("Trace: Log Posterior")
    axes[0, 2].set_xlabel("Iteration")
    axes[0, 2].set_ylabel("Log Posterior")
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Beta trace
    axes[1, 0].plot(beta_samples)
    if true_params and 'beta' in true_params:
        axes[1, 0].axhline(true_params['beta'], color='red', linestyle='--')
    axes[1, 0].set_title("Trace: Beta (Coherence)")
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_ylabel("Beta")
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Beta histogram
    axes[1, 1].hist(beta_samples, bins=30, density=True, alpha=0.7)
    if true_params and 'beta' in true_params:
        axes[1, 1].axvline(true_params['beta'], color='red', linestyle='--')
    axes[1, 1].set_title("Posterior: Beta")
    axes[1, 1].set_xlabel("Beta")
    axes[1, 1].set_ylabel("Density")
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Noise trace
    axes[1, 2].plot(sigma_samples)
    axes[1, 2].set_title("Trace: Noise (σ)")
    axes[1, 2].set_xlabel("Iteration")
    axes[1, 2].set_ylabel("Noise σ")
    axes[1, 2].grid(True, alpha=0.3)
    
    # 7. Fit quality - show data vs model for first q value
    q_idx = 0
    axes[2, 0].scatter(t_values, data[q_idx, :], alpha=0.5, s=10, label='Data')
    
    # Get posterior mean model
    posterior_mode_k = unique_k[jnp.argmax(counts)]
    
    # Find a state with modal k from snapshots
    for state in samples['state_snapshots'][-10:]:  # Check last 10 snapshots
        if state.k == posterior_mode_k:
            active_amp = state.amp[:state.k]
            active_mu = state.mu[:state.k]
            active_sig = state.sig[:state.k]
            
            model_pred = g2_minus1_matrix(q_values, t_values, active_amp, active_mu, active_sig, state.beta)
            axes[2, 0].plot(t_values, model_pred[q_idx, :], 'r-', alpha=0.8, label=f'Model (k={state.k})')
            break
    
    axes[2, 0].set_xscale('log')
    axes[2, 0].set_title(f"Fit Quality (q={q_values[q_idx]:.3f})")
    axes[2, 0].set_xlabel("Time")
    axes[2, 0].set_ylabel("g2(t) - 1")
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # 8. Residuals for all q values
    if len(samples['state_snapshots']) > 0:
        state = samples['state_snapshots'][-1]  # Use last state
        active_amp = state.amp[:state.k]
        active_mu = state.mu[:state.k]
        active_sig = state.sig[:state.k]
        
        model_pred = g2_minus1_matrix(q_values, t_values, active_amp, active_mu, active_sig, state.beta)
        residuals = data - model_pred
        
        im = axes[2, 1].imshow(residuals, aspect='auto', cmap='RdBu_r', vmin=-3*state.sigma_noise, vmax=3*state.sigma_noise)
        axes[2, 1].set_title("Residuals Map")
        axes[2, 1].set_xlabel("Time Index")
        axes[2, 1].set_ylabel("q Index")
        plt.colorbar(im, ax=axes[2, 1])
    
    # 9. Parameter summary text
    axes[2, 2].axis('off')
    summary_text = "Posterior Summary\n" + "="*20 + "\n\n"
    summary_text += f"k mode: {posterior_mode_k}\n"
    summary_text += f"k mean: {jnp.mean(k_samples):.2f}\n\n"
    
    for k_val, prob in sorted(zip(unique_k, posterior_probs), key=lambda x: -x[1])[:5]:
        summary_text += f"P(k={k_val}) = {prob:.3f}\n"
    
    summary_text += f"\nβ mean: {jnp.mean(beta_samples):.3f}\n"
    summary_text += f"β std: {jnp.std(beta_samples):.3f}\n"
    summary_text += f"\nσ mean: {jnp.mean(sigma_samples):.4f}\n"
    summary_text += f"σ std: {jnp.std(sigma_samples):.4f}\n"
    
    axes[2, 2].text(0.1, 0.9, summary_text, transform=axes[2, 2].transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/rjmcmc_{index:04d}.png", dpi=300)

def process_observation(index, output_dir="results", n_iter=20_000, burn_in=18_500):
    key = jax.random.PRNGKey(42 + index)
    
    i = index
    observations = clean_obs_list[i]
    valid_params_i = valid_params[i]
    amp, mu, sig = valid_params_i
    true_k = len(amp)
    true_params = {
        'k': true_k,
        'beta': gen_beta,
        'amp': amp,
        'mu': mu,
        'sig': sig,
        'sigma': noise_errors[i]
    }


    # Set priors
    k_max = 5
    prior_params = {
        'amp_alpha': 1.0,           # Dirichlet concentration
        'log_mu_mean': jnp.log(0.5),  # Log-normal prior for mu
        'log_mu_std': 0.5,
        'sig_alpha': 2.0,           # Beta parameters for sig
        'sig_beta': 2.0,
        'beta_alpha': 5.0,          # Beta prior for coherence factor
        'beta_beta': 2.0,           # Favors higher beta values
        'noise_a': 2.0,             # Inverse gamma for noise
        'noise_b': 0.0001,
    }

    # Initialize
    initial_k = 1
    initial_amp = jnp.zeros(k_max).at[:initial_k].set(1.0)
    initial_mu = jnp.zeros(k_max).at[:initial_k].set(0.7)
    initial_sig = jnp.zeros(k_max).at[:initial_k].set(0.09)
    initial_beta = 0.7
    initial_sigma = 0.01

    print(f"Data shape: {observations.shape}")
    print(f"True parameters: k={true_params['k']}, beta={true_params['beta']:.3f}")
    print(f"True mu: {true_params['mu']}")
    print(f"True sig: {true_params['sig']}")

    initial_state = DLSState(
        k=initial_k,
        amp=initial_amp,
        mu=initial_mu,
        sig=initial_sig,
        beta=initial_beta,
        sigma_noise=initial_sigma,
        log_posterior=0.0
    )

    # Run RJMCMC
    print(f"\nRunning RJMCMC for {n_iter} iterations...")
    key, sampler_key = random.split(key)
    # samples = rjmcmc_sampler_dls_bd(
    samples = rjmcmc_sampler_dls_paper(
        key=sampler_key,
        n_iter=n_iter,
        initial_state=initial_state,
        data=observations,
        q_values=q,
        t_values=t,
        k_max=k_max,
        prior_params=prior_params,
        verbose=True
    )

    # Plot results

    Path(output_dir).mkdir(exist_ok=True)
    result = {
        'index': i,
        'samples_param_snapshots': samples['state_snapshots']
    }
    
    with open(f'{output_dir}/res_rjmcmc_{i:04d}.pkl', 'wb') as f:
        pickle.dump(result, f)

    plot_dls_results(samples, observations, q, t, output_dir, index, burn_in=burn_in, true_params=true_params)

    # Print summary
    k_samples = samples['k'][burn_in:]
    unique_k, counts = jnp.unique(k_samples, return_counts=True)
    posterior_probs = counts / len(k_samples)

    print("\n" + "="*50)
    print("POSTERIOR SUMMARY")
    print("="*50)
    print(f"\nPosterior probabilities for k:")
    for k_val, prob in sorted(zip(unique_k, posterior_probs), key=lambda x: -x[1]):
        stars = "*" if k_val == true_params['k'] else " "
        print(f"  P(k={k_val} | data) = {prob:.4f} {stars}")

    print(f"\nBeta posterior: {jnp.mean(samples['beta'][burn_in:]):.3f} ± {jnp.std(samples['beta'][burn_in:]):.3f}")
    print(f"True beta: {true_params['beta']:.3f}")

    print(f"\nNoise posterior: {jnp.mean(samples['sigma_noise'][burn_in:]):.4f} ± {jnp.std(samples['sigma_noise'][burn_in:]):.4f}")
    print(f"True noise: {true_params['sigma']:.4f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python process_single_observation.py <index>")
        sys.exit(1)
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
    
    index = int(sys.argv[1])
    process_observation(index)










# ## generate data 
# key, model_key, data_key = jax.random.split(key, 3)

# q_vals, t_vals, _ = prep_data_g2("exp_data/stock_100nm.csv")
# # True parameters for generating synthetic data
# true_params = {
#     'k': 2,
#     'beta': 1.0,
#     'amp': jnp.array([0.5, 0.5]),
#     'mu': jnp.array([1.4, 0.3]),
#     'sig': jnp.array([0.2, 0.09]),
#     'sigma': 0.0005
#     # 'sigma': 0.0
# }
# true_g2 = g2_minus1_matrix(
#     q_vals, t_vals,
#     true_params['amp'], true_params['mu'], true_params['sig'], true_params['beta']
# )

# data = true_g2 + true_params['sigma'] * jax.random.normal(data_key, shape=true_g2.shape)


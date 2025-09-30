import jax
import jax.numpy as jnp
import orthax
import numpy as np

from hnmf_tr_optimizer.clusts import result_analysis

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

# for drawing normal curves
def normal_distribution_single(x, amplitude, mu, sigma):
    return amplitude * jnp.exp(-(x-mu)**2/(2*sigma**2))/jnp.sqrt(2*jnp.pi*sigma**2)

normal_distributions = jax.vmap(normal_distribution_single, in_axes=(None, 0, 0, 0))

def normal_distribution(possible_D, amp, mu, sig):
    whole = normal_distributions(possible_D, amp, mu, sig).sum(axis=0)
    return whole / jnp.sum(whole) # normalize for plotting


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

def g2_minus1_matrix(q, t, amp, mu, sig, beta):
    g1 = g1_matrix(q, t, amp, mu, sig)
    g2_minus1 = beta * g1**2
    return g2_minus1

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

def g2_minus1_matrix_dirac(q, t, D, amp, beta, const):
    amp = amp / jnp.sum(amp)
    g1 = g1_dirac(q, t, D, amp, const)
    g2_minus1 = beta * g1**2
    return g2_minus1


def gen_bounds_dirac(k):
    return (1e-9*jnp.ones(k),1e-9*jnp.ones(k),jnp.array([0.0])), (1e-3*jnp.ones(k), jnp.ones(k),jnp.array([1.0]))



# %%
# quadrature based model

def g1_quadrature_(q, t, amp, mu, sig, scaling_const, laguerre_deg):
    x_lag, w_lag = orthax.laguerre.laggauss(laguerre_deg)
    f_lag = normal_distributions(scaling_const * x_lag, amp, mu*SCALING_CONST, sig*SCALING_CONST)
    return scaling_const * jnp.sum(w_lag * f_lag * jnp.exp((1-scaling_const*q**2*t) * x_lag))

g1_quadrature_by_t = jax.vmap(g1_quadrature_, in_axes = (0, None, None, None, None, None, None))
g1_quadrature_by_q = jax.vmap(g1_quadrature_by_t, in_axes = (None, 0, None, None, None, None, None))

def g2_minus1_quadrature(q, t, amp, mu, sig, beta, scaling_const, laguerre_deg):
    # amp = amp/jnp.sum(amp)
    return beta*jnp.square(g1_quadrature_by_q(q, t, amp, mu, sig, scaling_const, laguerre_deg).T)





def gen_bounds(num_sources):
    lower_bounds = (
        1e-12*np.ones(num_sources),
        1e-12*np.ones(num_sources),
        1e-12*np.ones(num_sources),
    )
    upper_bounds = (
        np.inf*np.ones(num_sources),
        np.inf*np.ones(num_sources),
        np.inf*np.ones(num_sources),
    )
    return lower_bounds, upper_bounds

class InitParamsGenerator:
    def __init__(self, seed=42):
        self.key = jax.random.PRNGKey(seed)
    
    def __call__(self, num_sources):
        self.key, subkey1, subkey2, subkey3 = jax.random.split(self.key, 4)
        return (
            (1.0 - 0.1) * jax.random.uniform(subkey1, (num_sources,)) + 0.1,
            (15.0 - 1.5) * jax.random.uniform(subkey2, (num_sources,)) + 1.5,
            (1.5 - 0.1) * jax.random.uniform(subkey3, (num_sources,)) + 0.1,
        )

class InitParamsGenerator2:
    def __init__(self, bound_generator, seed=42):
        self.bound_generator = bound_generator
        self.key = jax.random.PRNGKey(seed)
    
    def __call__(self, num_sources):
        lb, ub = self.bound_generator(num_sources)
        num_params = len(lb)
        init = []
        for i in range(num_params):
            self.key, subkey = jax.random.split(self.key)
            lb_i = jnp.array(lb[i])
            ub_i = jnp.array(ub[i])
            generated = jax.random.uniform(subkey, lb_i.shape, minval=lb_i, maxval=ub_i)
            unbounded_upper_gen = jax.random.uniform(subkey, lb_i.shape, minval=lb_i)
            unbounded_lower_gen = jax.random.uniform(subkey, lb_i.shape, maxval=ub_i)
            generated = jnp.where(
                jnp.isposinf(generated),
                unbounded_upper_gen,
                generated
            )
            generated = jnp.where(
                jnp.isneginf(generated),
                unbounded_lower_gen,
                generated
            )
            init.append(generated)
        return tuple(init)






# small_particle_bound = 1e-10 # 1 angstrom
# large_particle_bound = 1e-5 # 10 microns
def gen_bounds_std(num_sources):
    lower_bounds = (
        1e-9*jnp.ones(num_sources),
        jnp.array([1e-9])
    )
    upper_bounds = (
        1e-3*jnp.ones(num_sources),
        jnp.array([1.0])
    )
    return lower_bounds, upper_bounds

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


def extract_point(row):
    # extract source amplitudes and positions to use as points for clustering
    points = []
    sol = row['sol']
    D = sol[0]
    amp = sol[1]
    beta = sol[2]
    if isinstance(amp, float):
        D = jnp.array([D])
        amp = jnp.array([amp])
        beta = jnp.array([beta])
    else:
        D = jnp.array(D)
        amp = jnp.array(amp)
        beta = jnp.array(beta)
    amp = amp/jnp.sum(amp)
    for p in range(len(amp)):
        point = jnp.stack([D[p], amp[p], beta[p]]).tolist()
        points.append(point)
    return points

def clustering_preprocess(res):
    res = res.copy().groupby('num_sources', group_keys=False)[res.columns.tolist()].apply(filter_quantile)
    res['points'] = res.apply(extract_point, axis=1)
    return res

def filter_quantile(res, col_to_filter='fval', quantile=0.25):
    mod_col = res[col_to_filter].apply(lambda x: jnp.inf if jnp.isnan(x) else x)
    return res[
        mod_col < mod_col.quantile(q=quantile)
    ]

def process_res_dirac_(all_res, obs_size):
    Forclusts = clustering_preprocess(all_res)
    Forclusts = Forclusts.groupby('num_sources', group_keys=False)[Forclusts.columns.tolist()].apply(lambda group: result_analysis(
        group['points'].sum(),
        group['normF'].mean(),
        obs_size,
        group['num_sources'].iloc[0]
    ))
    Forclusts = Forclusts.set_index('num_source')

    return Forclusts



def extract_point_std(row):
    # extract source amplitudes and positions to use as points for clustering
    points = []
    sol = row['sol']
    sig = sol[0]
    beta = sol[1]
    beta = jnp.array(beta)
    if isinstance(sig, float):
        sig = jnp.array([sig])
    else:
        sig = jnp.array(sig)
    for p in range(len(sig)):
        point = sig.reshape(-1, 1)[p].tolist()
        points.append(point)
    return points

def clustering_preprocess_std(res):
    res = res.copy().groupby('num_sources', group_keys=False)[res.columns.tolist()].apply(filter_quantile)
    res['points'] = res.apply(extract_point_std, axis=1)
    return res

def process_res_std_(all_res, obs_size):
    Forclusts = clustering_preprocess_std(all_res)
    Forclusts = Forclusts.groupby('num_sources', group_keys=False)[Forclusts.columns.tolist()].apply(lambda group: result_analysis(
        group['points'].sum(),
        group['normF'].mean(),
        obs_size,
        group['num_sources'].iloc[0]
    ))
    Forclusts = Forclusts.set_index('num_source')

    return Forclusts

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
        group['normF'].mean(),
        obs_size,
        group['num_sources'].iloc[0]
    ))
    Forclusts = Forclusts.set_index('num_source')

    return Forclusts

def clustering_preprocess_std_normal(res):
    res = res.copy().groupby('num_sources', group_keys=False)[res.columns.tolist()].apply(filter_quantile)
    res['points'] = res.apply(extract_point, axis=1)
    return res

def process_res_normal_(all_res, obs_size):
    Forclusts = clustering_preprocess_std_normal(all_res)
    Forclusts = Forclusts.groupby('num_sources', group_keys=False)[Forclusts.columns.tolist()].apply(lambda group: result_analysis(
        group['points'].sum(),
        group['normF'].mean(),
        obs_size,
        group['num_sources'].iloc[0]
    ))
    Forclusts = Forclusts.set_index('num_source')

    return Forclusts


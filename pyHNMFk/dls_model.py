import jax
import jax.numpy as jnp
import numpy as np

def stagger(arr, slice_axis=0, stagger_axis=1):
    l = arr.shape[stagger_axis]
    staggered = []
    for i in range(l):
        ss = jnp.zeros_like(arr)
        val_slice = jnp.take(arr, i, axis=slice_axis)
        ind = jnp.take(jnp.arange(arr.size).reshape(arr.shape), i, axis=slice_axis)
        ss = jnp.put(ss, ind, val_slice, inplace=False)
        staggered.append(ss)
    return jnp.concatenate(staggered, axis=stagger_axis)



# def source3(q, t, amp, mu, sig):
#     const = 2.45e-10
#     nk = amp
#     sig = sig*const
#     mu = mu*const
#     a = mu/sig
#     b = a**2/2
#     c = q**2*sig
#     d = jnp.sqrt(2)
#     s_pi = jnp.sqrt(jnp.pi)
#     x = (-a + c*t)/d
#     y = jnp.where(
#         jnp.greater_equal(x, 5),
#         1/(x*s_pi),
#         jax.scipy.special.erfc(x)*jnp.exp(x**2)
#     )
#     e = (nk/2)*jnp.exp(-b)
#     return e*y

# by_t = jax.vmap(
#     source3,
#     in_axes=(None, 0, None, None, None)
# )

# by_src = jax.vmap(
#     source3,
#     in_axes=(None, None, 0, 0, 0)
# )

# by_angle = jax.vmap(
#     by_src,
#     in_axes=(0, None, None, None, None)
# )

# def observational_matrix(q, t, amp, mu, sig):
#     full = by_angle(q, t, amp, mu, sig)
#     full = jnp.sum(full, axis=1)
#     return full



def get_g(t, nk, a, c):
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
all_g = jax.vmap(
    get_g,
    in_axes=(0, None, None, None)
)

def source3(t, q, amp, mu, sig):
    const = 2.45e-10
    nk = amp
    sig = sig*const
    mu = mu*const
    a = mu/sig
    c = q**2*sig

    g = all_g(t, nk, a, c)
    return g

by_Xs = jax.vmap(
    source3,
    in_axes=(None, None, 0, 0, 0)
)

source_matrix = jax.vmap(
    by_Xs,
    in_axes=(None, 0, None, None, None)
)

def observational_matrix(q, t, amp, mu, sig):
    full = source_matrix(t, q, amp, mu, sig)
    full = jnp.sum(full, axis=1)
    return full

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


# def get_init_params(num_sources):
#     return (
#         (1.0-0.1)*np.random.rand(num_sources) + 0.1,
#         (55.0-1.5)*np.random.rand(num_sources) + 1.5,
#         (15.5-0.1)*np.random.rand(num_sources) + 0.1,
#     )

# preprocessing functions to run before clustering
def clustering_preprocess(res):
    res = res.copy().groupby('num_sources', group_keys=False)[res.columns.tolist()].apply(filter_quantile)
    res['points'] = res.apply(extract_point, axis=1)
    return res

def filter_quantile(res, col_to_filter='fval', quantile=0.25):
    mod_col = res[col_to_filter].apply(lambda x: jnp.inf if jnp.isnan(x) else x)
    return res[
        mod_col < mod_col.quantile(q=quantile)
    ]

def extract_point(row):
    # extract source amplitudes and positions to use as points for clustering
    points = []
    sol = row['sol']
    amp = sol[0]
    mu = sol[1]
    sig = sol[2]
    for p in range(len(amp)):
        point = jnp.stack([amp[p], mu[p], sig[p]]).tolist()
        points.append(point)        
    return points


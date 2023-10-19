import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

# green model functions using JAX

@partial(jax.jit, static_argnames=['length'])
def identity_vec(length):
    # returns vector that looks like [1 0 .. 0]
    return jnp.squeeze(jax.lax.slice(jnp.identity(length), (0, 0), (1, length)))

def green(a_s, x_s, t_s, x_d, t, U, D):
    # make matrix nt x dims
    t_offset = t-t_s
    xx_s = jax.lax.broadcast_in_dim(x_s, (x_s.shape[0], t.shape[0]), (0,)).T
    xx_d = jax.lax.broadcast_in_dim(x_d, (x_d.shape[0], t.shape[0]), (0,)).T
    tt = jax.lax.broadcast_in_dim(t_offset, (x_s.shape[0], t.shape[0]), (1,)).T

    xx = xx_d - xx_s - U*tt
    gaussians = (-xx**2)
    gaussians = gaussians/(4*D*tt)
    gaussians = jnp.exp(gaussians)

    dividend = jnp.prod(D)
    dividend = jnp.sqrt(dividend)
    dividend = dividend*(t_offset)
    dividend = 4*jnp.pi*dividend

    res = a_s*jnp.prod(gaussians, axis=1)/dividend
    return res

# run green for each detector
green_detector_mapped = jax.vmap(
    green,
    in_axes=(None, None, None, 0, None, None, None)
)

# run green_detector for each source 
green_matrix = jax.vmap(
    green_detector_mapped,
    in_axes=(0, 0, None, None, None, None, None)
)

# @jax.jit
def reconstruct_general(A_s, X_s, T_s, X_d, t, U, D):
    U = U * identity_vec(D.shape[0])
    return jnp.transpose(jnp.sum(green_matrix(A_s, X_s, T_s, X_d, t, U, D), axis=0))

# Helpers - input shaping, generating bounds and initial points

def gen_bounds(num_sources):
    # instead of 0, use epsilon of 1e-10
    lower_bounds = (
        1e-10*np.ones(num_sources),
        -1*np.ones((num_sources, 2)),
        1e-10*np.ones(1),
        1e-10*np.ones(2),
    )
    upper_bounds = (
        1.5*np.ones(num_sources),
        np.ones((num_sources, 2)),
        np.ones(1),
        np.ones(2),
    )
    return lower_bounds, upper_bounds

def gen_init_params(num_sources):
    return (
        np.random.rand(num_sources),
        2*(0.5 - np.random.rand(num_sources, 2)),
        np.random.rand(1),
        np.random.rand(2),
    )

# preprocessing functions to run before clustering
def clustering_preprocess(res):
    res = res.copy().groupby('num_sources', group_keys=False).apply(filter_quantile)
    res['points'] = res.apply(extract_point, axis=1)
    return res

def filter_quantile(res, col_to_filter='fval', quantile=0.25):
    return res[
        res[col_to_filter] < res[col_to_filter].quantile(q=quantile)
    ]

def extract_point(row):
    # extract source amplitudes and positions to use as points for clustering
    points = []
    sol = row['sol']
    a = sol[0] # amplitudes
    x = sol[1] # locations
    for p in range(len(a)):
        point = [a[p]]
        point.extend(x[p])
        point.extend(sol[2])
        point.extend(sol[3])
        point = jnp.stack(point).tolist()
        points.append(point)
        
    return points


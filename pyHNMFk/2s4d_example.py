# import jax.numpy as jnp
# import jax
# jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_debug_nans", True)
# import os
# import fides
# import pandas as pd
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score, silhouette_samples
import numpy as np
# import math
# import time
# import logging
# import graphviz
# from functools import partial
# import sys
# from matplotlib import pyplot as plt

from green_model import reconstruct_general, gen_init_params,  gen_bounds, clustering_preprocess

from src.hnmf_optimizer import HNMFOptimizer
from src.clusts import result_analysis
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#       set true variables for generated observations       #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

### parameters ###
#amplitudes of the sources
As = np.array([0.5, 0.7])

# # positions of the sources
Xs = np.array([
    [-0.1, -0.2],
    [-0.9, -0.8]
])


# Advection velocity
u_scaler = 0.05

# Dispersivity
D = np.array([0.005, 0.00125])

### fixed known values ###
# the initial time the sources begin the release
Ts = -10.

# positions of the detectors
Xd = np.array([
    [0, 0],
    [-0.5, -0.5],
    [0.5, 0.5],
    [0.5, -0.5]
])


### variables ####
# time values of data
t = np.linspace(0, 20, 80)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                  generate observations                    #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

noise_level = 1e-4
observations = reconstruct_general(As, Xs, Ts, Xd, t, np.array([u_scaler]), D)
noise = noise_level/2 - noise_level*np.random.rand(*observations.shape)
observations += noise





opt = HNMFOptimizer(
    model_fn = reconstruct_general,
    param_generator = gen_init_params,
    bound_generator = gen_bounds,
    input_args = ('t',),
    param_args = ('A_s', 'X_s', 'U', 'D'),
    constants = {'T_s': Ts, 'X_d': Xd},
    min_k = 1,
    max_k = 5,
    nsim = 100
)

res = opt(t, observations)

Forclusts = clustering_preprocess(res)

Forclusts = Forclusts.groupby('num_sources', group_keys=False).apply(lambda group: result_analysis(
    group['points'].sum(),
    group['normF'].mean(),
    observations.size,
    group['num_sources'].iloc[0]
))
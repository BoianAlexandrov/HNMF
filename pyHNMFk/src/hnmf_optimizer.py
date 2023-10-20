import functools
import inspect
import logging
import sys
import time

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)
from .trust_region_optimizer import TrustRegionOptimizer
import numpy as np
import pandas as pd

def flatten(*inputs):
    flat = []
    shapes = []
    for input in inputs:
        shapes.append(input.shape)
        flat.extend(input.flatten())
    return np.array(flat), tuple(shapes)

def unflatten(flat_inputs, shapes):
    i = 0
    inputs = []
    for shape in shapes:
        size = np.prod(shape)
        inputs.append(np.reshape(flat_inputs[i:i+size], shape))
        i+=size
    return inputs

# quicker calculation for val and jac
# source: https://github.com/google/jax/pull/762#issuecomment-1002267121
def value_and_jacfwd(f, x):
    pushfwd = functools.partial(jax.jvp, f, (x,))
    basis = jnp.eye(x.size, dtype=x.dtype)
    y, jac = jax.vmap(pushfwd, out_axes=(None, 1))((basis,))
    return y, jac

def value_and_jacrev(f, x):
    y, pullback = jax.vjp(f, x)
    basis = jnp.eye(y.size, dtype=y.dtype)
    jac = jax.vmap(pullback)(basis)
    return y, jac


class HNMFOptimizer:
    def __init__(
            self,
            model_fn,
            param_generator,
            bound_generator,
            input_args,
            param_args,
            constants,
            min_k,
            max_k,
            nsim
        ):
        """
        model_fn - residual function
        param_generator - takes k (# of sources) as input and returns generated/random paramaters
        bound_generator takes k as input and returns (lower_bounds, upper_bounds) of parameters
        """
        self.model_fn = model_fn
        self.param_generator = param_generator
        self.bound_generator = bound_generator
        self.input_args = input_args
        self.param_args = param_args
        self.constants = constants
        self.min_k = min_k
        self.max_k = max_k
        self.nsim = nsim
        self.flatten = flatten
        self.unflatten = unflatten

    def num_source2shapes(self, num_sources):
        sample_input = self.param_generator(num_sources)
        _, input_shapes = self.flatten(*sample_input)
        return input_shapes

    def make_resid_fn(self, k, inputs, observations):
        if not isinstance(inputs, tuple):
            inputs = (inputs,)
        all_described = list()
        all_described.extend(self.input_args)
        all_described.extend(self.param_args)
        all_described.extend(list(self.constants.keys()))
        fn_args = inspect.getfullargspec(self.model_fn).args
        assert all([arg in fn_args for arg in all_described])

        unflatten_params_fn = functools.partial(self.unflatten, shapes=self.num_source2shapes(k))

        # @jax.jit
        def flat_resid_fn(flat_params):
            params = unflatten_params_fn(flat_params)
            args_dict = self.constants.copy()
            for i, k in enumerate(self.input_args):
                args_dict[k] = inputs[i]
            for i, k in enumerate(self.param_args):
                args_dict[k] = params[i]
            return jnp.ravel(observations - self.model_fn(**args_dict))
        return flat_resid_fn

    def make_obj_func(self, k, inputs, observations):
        resid = self.make_resid_fn(k, inputs, observations)
        # @jax.jit
        def obj(x):
            r, jac_r = value_and_jacfwd(resid, x)
            loss = 0.5*jnp.sum(jnp.square(r))
            grad = jnp.matmul(r.T, jac_r)
            hess = jnp.matmul(jac_r.T, jac_r)
            return loss, grad, hess

        return obj

    def setup_optimizer(
            self,
            k,
            inputs,
            observations,
            log_level=logging.ERROR,
            opt_options=None
        ):
        obj = self.make_obj_func(k, inputs, observations)
        lb, ub = self.bound_generator(k)
        lb, _ = self.flatten(*lb)
        ub, _ = self.flatten(*ub)

        return TrustRegionOptimizer(
            obj,
            ub=ub,
            lb=lb,
            options=opt_options,
            verbose = log_level
        )

    def __call__(self, inputs, observations, opt_options=None):
        AA = 0 # some normalization factor to be used for AIC calculation later
        for i in range(observations.shape[1]):
            AA += np.sum(observations[:, i]**2)

        result_dfs = []
        errors = []
        for k in range(self.min_k, self.max_k+1):
            ### define optimization object ###
            t1 = time.time()
            opt = self.setup_optimizer(k, inputs, observations, opt_options=opt_options)

            ### run minimization on nsim random inits ###
            results = []
            successes = 0
            while successes < self.nsim:
                flat_init, _ = self.flatten(*self.param_generator(k))
                try:
                    res = opt.minimize(np.asarray(flat_init))
                    results.append(res)
                    successes+=1
                except: # TODO: catch specific exception types
                    the_type, the_value, the_traceback = sys.exc_info()
                    errors.append((the_type, the_value, the_traceback))
                    print(the_type)
            res = pd.DataFrame(columns=['fval', 'sol', 'grad', 'hess'], data=results)
            # norm from matlab HNMF code
            res['normF'] = np.sqrt((res['fval'].apply(float)/AA))*100
            res['num_sources'] = k
            
            result_dfs.append(res)

            t2 = time.time()
            print(f'SIMULATIONS FOR {k} SOURCES TOOK {t2-t1} SECONDS')        
        all_results = pd.concat(result_dfs)
        all_results['sol'] = all_results.apply(
            lambda row: self.unflatten(row['sol'], self.num_source2shapes(row['num_sources'])),
            axis=1
        )
        return all_results

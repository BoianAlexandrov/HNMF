import functools

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_debug_nans", True)
np_eps = jnp.finfo(jnp.float64).eps

def jax_multi_where(cond, res_true, res_false):
    return jax.lax.cond(
        cond,
        lambda _: res_true,
        lambda _: res_false,
        ()
    )

def normalize(v):
    nv = jnp.linalg.norm(v)
    return jax.lax.select(nv > 0, v/nv, v)

def get_affine_scaling(x, grad, lb, ub):
    """
    Computes the vector v and dv, the diagonal of its Jacobian. For the
    definition of v, see Definition 2 in [Coleman-Li1994]

    :return:
        v scaling vector
        dv diagonal of the Jacobian of v wrt x
    """
    # this implements no scaling for variables that are not constrained by
    # bounds ((iii) and (iv) in Definition 2)
    _v = jnp.sign(grad) + (grad == 0)
    _dv = jnp.zeros(x.shape)

    # this implements scaling for variables that are constrained by
    # bounds ( i and ii in Definition 2) bounds is equal to ub if grad <
    # 0 lb if grad >= 0
    bounds = jax.lax.select(grad < 0, ub, lb)
    bounded = jnp.isfinite(bounds)
    v = jnp.where(bounded, x - bounds, _v)
    dv = jnp.where(bounded, 1, _dv)
    return v, dv

def quadratic_form(Q, p, x):
    return 0.5 * x.T.dot(Q).dot(x) + p.T.dot(x)

def quadratic_form_shess(shess_func, p, x):
    return 0.5 * x.T.dot(shess_func(x)) + p.T.dot(x)

def slam(lam, w, eigvals, eigvecs):
    el = eigvals + lam
    c = jnp.where(el != 0, w/el, w)
    return eigvecs.dot(c)

def dslam(lam, w, eigvals, eigvecs):
    el = eigvals + lam
    _c = jnp.where(el != 0, w/-jnp.power(el, 2), w)
    c = jnp.where((el == 0) & (_c != 0), jnp.inf, _c)
    return eigvecs.dot(c)

def secular(lam,w,eigvals,eigvecs,delta):
    res1 = jax.lax.select(lam < -jnp.min(eigvals), jnp.inf, 0.)
    s = slam(lam, w, eigvals, eigvecs)
    sn = jnp.linalg.norm(s)
    res2 = jax.lax.select(sn > 0, 1 / sn - 1 / delta, jnp.inf)
    return (res1 + res2)

def dsecular(lam, w, eigvals, eigvecs, delta):
    s = slam(lam, w, eigvals, eigvecs)
    ds = dslam(lam, w, eigvals, eigvecs)
    sn = jnp.linalg.norm(s)
    return jax.lax.select(sn > 0, -s.T.dot(ds) / (jnp.linalg.norm(s) ** 3), jnp.inf)

def secular_and_grad(x, w, eigvals, eigvecs, delta):
    return (
        secular(x, w, eigvals, eigvecs, delta),
        dsecular(x, w, eigvals, eigvecs, delta)
    )

def secular_newton(x0, w, eigvals, eigvecs, delta, num_iter):
    """
    Newton's method for root-finding.
    no convergence criteria, just set number of iterations but if an iteration leads to inf/nan
    it is ignored and the following iterations essentially become expensive noops
    """
    def body(it, x):
        fx, dfx = secular_and_grad(x, w, eigvals, eigvecs, delta)
        step = fx / dfx
        new_x = x - step
        return jnp.where(jnp.isfinite(new_x), new_x, x)

    return jax.lax.fori_loop(
        0,
        num_iter,
        body,
        x0,
    )


def solve_nd_trust_region_subproblem_jitted(B, g, delta):
    # See Nocedal & Wright 2006 for details
    # INITIALIZATION

    def hard_case(w, mineig, eigvals, eigvecs, delta, jmin):
        w = jnp.where((eigvals - mineig) == 0, 0, w)
        s = slam(-mineig, w, eigvals, eigvecs)
        # we know that ||s(lam) + sigma*v_jmin|| = delta, since v_jmin is
        # orthonormal, we can just substract the difference in norm to get
        # the right length.

        sigma = jnp.sqrt(jnp.maximum(delta**2 - jnp.linalg.norm(s) ** 2, 0))
        s = s + sigma * eigvecs[:, jmin]
        # logger.debug('Found boundary 2D subproblem solution via hard case')
        return s

    # instead of a cholesky factorization, we go with an eigenvalue
    # decomposition, which works pretty well for n=2
    eigvals, eigvecs = jnp.linalg.eig(B)
    eigvals = jnp.real(eigvals)
    eigvecs = jnp.real(eigvecs)
    w = -eigvecs.T.dot(g)
    jmin = eigvals.argmin()
    mineig = eigvals[jmin]

    # since B symmetric eigenvecs V are orthonormal
    # B + lambda I = V * (E + lambda I) * V.T
    # inv(B + lambda I) = V * inv(E + lambda I) * V.T
    # w = V.T * g
    # s(lam) = V * w./(eigvals + lam)
    # ds(lam) = - V * w./((eigvals + lam)**2)
    # \phi(lam) = 1/||s(lam)|| - 1/delta
    # \phi'(lam) = - s(lam).T*ds(lam)/||s(lam)||^3
    laminit = jax.lax.select(mineig > 0, 0.0, -mineig)

    # calculate s for positive definite case
    s = jnp.real(slam(0, w, eigvals, eigvecs))
    norm_s = jnp.linalg.norm(s)
    thresh = delta + jnp.sqrt(np_eps)
    posdef_cond = jnp.logical_and((mineig > 0), (norm_s <= thresh))
    neg_sval = secular(laminit, w, eigvals, eigvecs, delta) < 0


    maxiter = 100
    root = secular_newton(laminit, w, eigvals, eigvecs, delta, maxiter)
    indef_s = slam(root, w, eigvals, eigvecs)
    is_root = jnp.linalg.norm(indef_s) <= delta + 1e-12
    indef = jnp.logical_and(neg_sval, is_root)

    other_s = jax.lax.cond(
        indef,
        lambda *_: indef_s,
        hard_case,
        w, mineig, eigvals, eigvecs, delta, jmin
    )
    other_case = jax.lax.select(indef, 1, 2)
    

    s = jax.lax.select(posdef_cond, s, other_s)
    hess_case = jax.lax.select(posdef_cond, 0, other_case)
    return s, hess_case

def step_compute(x, subspace, sg, shess_func, delta, lb, ub, scaling, ss0, theta):
    ### project to subspace ###
    chess = subspace.T.dot(jax.vmap(shess_func, 1)(subspace).T)
    cg = subspace.T.dot(sg)

    ### compute step ###
    sc, _ = solve_nd_trust_region_subproblem_jitted(
        chess,
        cg,
        jnp.sqrt(jnp.maximum(delta**2 - jnp.linalg.norm(ss0) ** 2, 0.0)),
    )
    ss = subspace.dot(jnp.real(sc))
    s = scaling.dot(ss)

    ### step back ###
    # create copies of the calculated step
    og_s = s.copy()
    og_ss = ss.copy()
    og_sc = sc.copy()

    # br quantifies the distance to the boundary normalized
    # by the proposed step, this indicates the fraction of the step
    # that would put the respective variable at the boundary
    # This is defined in [Coleman-Li1994] (3.1)
    nonzero = jnp.abs(s) > 0
    br = jnp.where(
        nonzero,
        jnp.max(jnp.vstack([(ub - x) / s,(lb - x) / s,]),axis=0),
        jnp.inf * jnp.ones(s.shape)
    )

    minbr = jnp.min(br)
    iminbr = jnp.argmin(br)

    # compute the minimum of the step
    alpha = jnp.min(jnp.array([1, theta * minbr]))

    s = s * alpha
    sc = sc * alpha
    ss = ss * alpha

    qpval = quadratic_form_shess(shess_func, sg, ss + ss0)

    return (s, ss, sc, og_s, og_ss, og_sc, qpval, br, iminbr, alpha)


def tr_iteration(x, grad, hvp, lb, ub, theta_max, delta):
    v, dv = get_affine_scaling(x, grad, lb, ub)

    theta = jnp.maximum(theta_max, 1 - jnp.linalg.norm(v * grad, jnp.inf))
    scale_vec = jnp.sqrt(jnp.abs(v))
    scaling = jnp.diag(scale_vec)

    sg = scale_vec*grad
    g_dscaling_vec = jnp.abs(grad) * dv

    def lin_op_(g_dscaling_vec, scale_vec, xx):
        sx = scale_vec*xx
        almost_done = hvp(x, sx)
        scaled = scale_vec*almost_done
        return scaled + g_dscaling_vec*xx

    # find the gauss-newton optimal point without matrix multiplication
    shess_func = functools.partial(lin_op_, g_dscaling_vec, scale_vec)
    og_s_newt = -jax.scipy.sparse.linalg.cg(shess_func, sg)[0]

    s0 = jnp.zeros(sg.shape)
    ss0 = jnp.zeros(sg.shape)
    s_newt_ = normalize(og_s_newt)
    subspace_0 = jnp.vstack([s_newt_, jnp.zeros(s_newt_.shape)]).T
    s_newt = s_newt_
    s_grad = sg.copy()
    s_newt = normalize(s_newt)
    s_grad = s_grad - s_newt * s_newt.dot(s_grad)
    subspace_other = jax.lax.select(
        jnp.linalg.norm(s_grad) > np_eps,
        jnp.vstack([s_newt, normalize(s_grad)]).T,
        jnp.vstack([s_newt, jnp.zeros(s_newt.shape)]).T
    )

    subspace = jax.lax.select(jnp.linalg.norm(og_s_newt) < delta, subspace_0, subspace_other)
    
    s, ss, sc, og_s, og_ss, og_sc, qpval, br, iminbr, alpha = step_compute(
        x, subspace, sg, shess_func, delta, lb, ub, scaling, ss0, theta
    )

    ### TRT step ###
    trt_s0 = s0.at[iminbr].set(s0[iminbr] + theta * br[iminbr] * og_s[iminbr])
    trt_ss0 = ss0.at[iminbr].set(ss0[iminbr] + theta * br[iminbr] * og_ss[iminbr])
    # update x and at breakpoint
    trt_x = x + trt_s0

    trt_subspace = subspace.at[iminbr, :].set(0)
    #  normalize subspace
    for ix in range(trt_subspace.shape[1]):
        # column normalization
        trt_subspace.at[:, ix].set(normalize(trt_subspace[:, ix]))
   # trt_subspace = jax.vmap(normalize, 1)(trt_subspace).T
 
    def dummy(trt_x, trt_subspace, sg, delta, lb, ub, scaling, trt_ss0, theta):
        return step_compute(trt_x, trt_subspace, sg, shess_func, delta, lb, ub, scaling, trt_ss0, theta)

    trt_s, trt_ss, trt_sc, trt_og_s, trt_og_ss, trt_og_sc, trt_qpval, _, _, _ = jax.lax.cond(
        alpha < 1.0,
        dummy,
        lambda *_: (s, ss, sc, og_s, og_ss, og_sc, jnp.inf, br, iminbr, alpha),
        trt_x, trt_subspace, sg, delta, lb, ub, scaling, trt_ss0, theta
    )

    s, ss, sc, og_s, og_ss, og_sc, qpval, step_type = jax_multi_where(
        jnp.logical_and(alpha < 1.0, trt_qpval < qpval),
        (trt_s, trt_ss, trt_sc, trt_og_s, trt_og_ss, trt_og_sc, trt_qpval, 0),
        (s, ss, sc, og_s, og_ss, og_sc, qpval, 1)
    )

    x_new = x + s

    return {
        'x_new': x_new,
        'qpval': qpval,
        'dv': dv,
        's': s,
        's0': s0,
        'ss': ss,
        'ss0': ss0,
        'type': step_type
    }

#### Optimizer ####

class TrustRegionOptimizer:
    def __init__(self, obj_fn, hvp, lb = None, ub = None, **kwargs):
        init_kwargs = kwargs
        if kwargs.get('options'):
            init_kwargs.update(kwargs['options'])
        if not lb is None:
            init_kwargs['lb'] = lb
        if not ub is None:
            init_kwargs['ub'] = ub
        self.init_kwargs = init_kwargs
        self.obj_fn = jax.jit(obj_fn)
        self.hvp = hvp
        self.state = None
        self.converge_cond = lambda state: state['finished']

        def update(state):
            # hvp = functools.partial(self.hvp, state['x'])
            step = tr_iteration(
                state['x'],
                state['grad'],
                self.hvp,
                state['lb'],
                state['ub'],
                state['theta_max'],
                state['delta']
            )
            state['x_sol'] = step['x_new']
            state['dv'] = step['dv']
            state['qpval'] = step['qpval']
            state['type'] = step['type']
            state['s'] = step['s']
            state['s0'] = step['s0']
            state['ss'] = step['ss']
            state['ss0'] = step['ss0']

            state['iter'] = state['iter'] + 1

            # check next step for acceptance and update radius
            loss, grad= obj_fn(state['x_sol'])
            curr_delta = state['delta']
            state['stepsx'] = state['ss'] + state['ss0']
            state['nsx'] = jnp.linalg.norm(state['stepsx'])
            state['normdx'] = jnp.linalg.norm(state['s'] + state['s0'])
            state['f_diff'] = jnp.abs(loss - state['fval'])

            def infinite_case(state, *args):
                state['tr_ratio'] = 0.0
                state['delta'] = jnp.nanmin(
                    jnp.array([state['delta'] * state['gamma1'], state['nsx'] / 4])
                )
                state['accepted'] = False
                return state

            def finite_case(state, loss, grad, curr_delta):
                # state, loss, grad, curr_delta = input_tuple
                aug = 0.5 * jnp.dot(state['stepsx'], state['dv'] * jnp.abs(grad) * state['stepsx'])
                actual_decrease = state['fval'] - loss - aug
                predicted_decrease = -state['qpval']
                state['tr_ratio'] = jnp.where(predicted_decrease <= 0.0, 0.0, actual_decrease/predicted_decrease)
                increse_cond = jnp.logical_and(
                    jnp.greater_equal(state['tr_ratio'], state['eta']),
                    jnp.logical_not(jnp.less(state['nsx'], curr_delta * 0.9))
                )
                decrease_cond = jnp.less_equal(state['tr_ratio'], state['mu'])
                skip_cond = jnp.logical_and(
                    jnp.less(state['mu'], state['tr_ratio']),
                    jnp.less(state['tr_ratio'], state['eta'])
                )
                ind = jnp.argmax(jnp.array([increse_cond, decrease_cond, skip_cond]))
                state['delta'] = jax.lax.switch(
                    ind,
                    [
                        lambda state: state['gamma2'] * state['delta'],
                        lambda state: jnp.nanmin(jnp.array([state['delta'] * state['gamma1'], state['nsx'] / 4])),
                        lambda state: state['delta'],
                    ],
                    state
                )

                state['accepted'] = state['tr_ratio'] > 0.0
                return state

            state = jax.lax.cond(
                jnp.isfinite(loss),
                finite_case,
                infinite_case,
                state, loss, grad, curr_delta
            )

            state['f_old'], state['x'], state['fval'], state['grad'], state['gnorm'] = jax_multi_where(
                state['accepted'],
                (state['fval'], state['x_sol'], loss, grad, jnp.linalg.norm(state['grad'])),
                (state['f_old'], state['x'], state['fval'], state['grad'], state['gnorm'])
            )

            state['converged'] = jnp.any(jnp.array([
                jnp.logical_and(
                    jnp.greater(state['tr_ratio'], state['mu']),
                    jnp.less(state['f_diff'], state['fatol'] + state['frtol'] * state['f_old'])
                ),
                jnp.logical_and(jnp.greater(state['iter'], 1), jnp.less(state['nsx'], state['xtol'])),
                jnp.less_equal(state['gnorm'], state['gatol']),
                jnp.less_equal(state['gnorm'], state['grtol'] * jnp.abs(state['f_old']))
            ]))

            state['finished'] = jnp.any(jnp.array([
                state['converged'],
                jnp.greater_equal(state['iter'], state['maxiter']),
                jnp.less_equal(state['delta'], np_eps)
            ]))

            return state
        self.update = jax.jit(update)

    def minimize(self, params):
        state = self.init_state(params, **self.init_kwargs)
        while not self.converge_cond(state):
            state = self.update(state)
            # self.log_step(state)
        self.state = state
        return (
            state['fval'],
            state['x'],
            state['grad'],
        )

    def init_state(self, params, **kwargs):
        loss, grad= self.obj_fn(params)
        return {
            'x': params,
            'fval': loss,
            'grad': grad,
            'gnorm': jnp.linalg.norm(grad),
            'iter': 0,
            # optimizer params
            'lb': kwargs.get('lb') if kwargs.get('lb') is not None else -jnp.inf*jnp.ones(params.shape),
            'ub': kwargs.get('ub') if kwargs.get('ub') is not None else jnp.inf*jnp.ones(params.shape),
            'maxiter': kwargs.get('maxiter') or 1500,
            'fatol': kwargs.get('fatol') or 1e-8,
            'frtol': kwargs.get('frtol') or 1e-8,
            'xtol': kwargs.get('xtol') or 0.0,
            'gatol': kwargs.get('gatol') or 1e-6,
            'grtol': kwargs.get('grtol') or 0.0,
            'theta_max': kwargs.get('theta_max') or 0.95,
            'mu': kwargs.get('mu') or 0.25,
            'eta': kwargs.get('eta') or 0.75,
            'gamma1': kwargs.get('gamma1') or 0.25,
            'gamma2': kwargs.get('gamma2') or 2.0,
            # step values
            'x_sol': jnp.nan,
            'f_old': loss,
            'f_diff': 0.0,
            'delta': kwargs.get('delta') or 1.0,
            'tr_ratio': 0.0,
            'dv': jnp.nan,
            'qpval': jnp.nan,
            'type': -1,
            's': jnp.nan,
            's0': jnp.nan,
            'ss': jnp.nan,
            'ss0': jnp.nan,
            'stepsx': jnp.nan,
            'nsx': jnp.nan,
            'normdx': jnp.nan,
            'accepted': False,
            'converged': False,
            'finished': False,
        }

    def log_step(self, state):
        if state['iter'] % 10 == 0:
            print(
                '     iter'
                '|    fval   |   fdiff  | tr ratio '
                '|tr radius|  ||g||  | ||step||| step|acc'
            )
        print(
            f'       {state["iter"]}'
            f'| {state["fval"]:+.2E} '
            f'| {state["f_diff"]:+.1E} '
            f'| {state["tr_ratio"]:+.1E} '
            f'| {state["delta"]:.1E} '
            f'| {state["gnorm"]:.1E} '
            f'| {state["normdx"]:.1E} '
            f'|   {state["type"]} '
            f'| {state["accepted"]}'
        )

class ParallelTrustRegionOptimizer(TrustRegionOptimizer):
    def __init__(self, obj_fn, lb = None, ub = None, **kwargs):
        super(ParallelTrustRegionOptimizer, self).__init__(obj_fn, lb = None, ub = None, **kwargs)
        self.pupdate = jax.pmap(self.update)
        self.pcond = jax.pmap(self.converge_cond)
    
    def init_state(self, params, **kwargs):
        states = [super(ParallelTrustRegionOptimizer, self).init_state(single_params, **kwargs) for single_params in params]
        pstate = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *states)
        return pstate

    def pop_replace_ind(self, pstate, ind, new_params, **kwargs):
        # retrieve individual state
        res_state = jax.tree_util.tree_map(lambda x: x[ind], pstate)
        # replace individual state
        new_state = super(ParallelTrustRegionOptimizer, self).init_state(new_params, **kwargs)
        pstate = jax.tree_util.tree_map(lambda x, y: x.at[ind].set(y), pstate, new_state)
        return res_state, pstate

    def minimize(self, params):
        num_devices = jax.device_count()
        pstate = self.init_state(params[:num_devices], **self.init_kwargs)
        results = []
        param_ind = num_devices
        while len(results) < len(params):
            # done = jnp.where(state['finished'])
            # print(f'done: {done}')
            # for ind in done:
            if jnp.any(pstate['finished']):
                for ind in range(pstate['finished'].shape[0]):
                    if pstate['finished'][ind]:
                        # print(f'state converge before: {pstate["finished"]}')
                        res_state, pstate = self.pop_replace_ind(pstate, ind, params[param_ind],  **self.init_kwargs)
                        # print(f'state converge after: {pstate["converged"]}')
                        results.append(res_state)
                        # hacky way to complete all params
                        # TODO: handle in cleaner way
                        param_ind = min(param_ind + 1, params.shape[0]-1)
            pstate = self.pupdate(pstate)
        # print(results)
        # return results
        return [
            (state['fval'], state['x'], state['grad'])
            for state in results
        ]

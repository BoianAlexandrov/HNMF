import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)
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

def copysign(a, b):
    return jnp.abs(-a)*(jnp.sign(b) + (b == 0))

def get_1d_trust_region_boundary_solution(B, g, s, s0, delta):
    a = jnp.dot(s, s)
    b = 2 * jnp.dot(s0, s)
    c = jnp.dot(s0, s0) - delta**2

    aux = b + copysign(jnp.sqrt(b**2 - 4 * a * c), b)
    ts = jnp.array([-aux / (2 * a), -2 * c / aux])

    qf = jax.vmap(quadratic_form, in_axes=(None, None, 0))
    qs = qf(B, g, s0 + jnp.outer(ts, s))

    return ts[jnp.argmin(qs)]

def solve_1d_trust_region_subproblem(B, g, s, delta, s0):
    """
    Solves the special case of a one-dimensional subproblem

    :param B:
        Hessian of the quadratic subproblem
    :param g:
        Gradient of the quadratic subproblem
    :param s:
        Vector defining the one-dimensional search direction
    :param delta:
        Norm boundary for the solution of the quadratic subproblem
    :param s0:
        reference point from where search is started, also counts towards
        norm of step

    :return:
        Proposed step-length
    """
    a = 0.5 * B.dot(s).dot(s)
    b = s.T.dot(B.dot(s0) + g)

    minq = -b / (2 * a)

    bound_cond = jnp.logical_and(a > 0, jnp.linalg.norm(minq * s + s0) <= delta)
    tau = jax.lax.cond(
        bound_cond,
        lambda *_: minq,
        get_1d_trust_region_boundary_solution,
        B, g, s, s0, delta
    )


    res = tau * jnp.ones((1,))
    null_res = jnp.zeros_like(res)

    return jax.lax.select(
        jnp.logical_and(delta == 0.0, jnp.array_equal(s, jnp.zeros_like(s))),
        null_res,
        res
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

def step_compute(x, subspace, sg, shess, delta, lb, ub, scaling, ss0, theta):
    ### project to subspace ###
    chess = subspace.T.dot(shess.dot(subspace))
    cg = subspace.T.dot(sg)

    ### compute step ###
    sc_nd, _ = solve_nd_trust_region_subproblem_jitted(
        chess,
        cg,
        jnp.sqrt(jnp.maximum(delta**2 - jnp.linalg.norm(ss0) ** 2, 0.0)),
    )
    sc_1 = solve_1d_trust_region_subproblem(shess, sg, subspace[:, 0], delta, ss0)
    sc_1d = jnp.zeros_like(sc_nd).at[0].set(1) * sc_1

    sc = jax.lax.select(jnp.linalg.matrix_rank(subspace) == 1, sc_1d, sc_nd)

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

    qpval = quadratic_form(shess, sg, ss + ss0)

    return (s, ss, sc, og_s, og_ss, og_sc, qpval, br, iminbr, alpha)


def tr_iteration(x, grad, hess, lb, ub, theta_max, delta):
    v, dv = get_affine_scaling(x, grad, lb, ub)

    ### trust region init ###

    scaling = jnp.diag(jnp.sqrt(jnp.abs(v)))
    theta = jnp.maximum(theta_max, 1 - jnp.linalg.norm(v * grad, jnp.inf))

    sg = scaling.dot(grad)
    # diag(g_k)*J^v_k Eq (2.5) [ColemanLi1994]
    g_dscaling = jnp.diag(jnp.abs(grad) * dv)


    ### step ###
    # B_hat (Eq 2.5) [ColemanLi1996]
    shess = jnp.matmul(jnp.matmul((scaling), hess), (scaling)) + g_dscaling

    s0 = jnp.zeros(sg.shape)
    ss0 = jnp.zeros(sg.shape)

    ### 2D steps ###

    og_s_newt = -jnp.linalg.lstsq(shess, sg)[0]
    # lstsq only returns absolute ev values
    e, v_ = jnp.linalg.eig(shess)
    posdef = jnp.min(jnp.real(e)) > -np_eps * jnp.max(jnp.abs(e))

    s_newt_ = normalize(og_s_newt)
    subspace_0 = jnp.vstack([s_newt_, jnp.zeros(s_newt_.shape)]).T


    s_newt_2 = jnp.real(v_[:, jnp.argmin(jnp.real(e))])
    s_newt = jax.lax.select(posdef, s_newt_, s_newt_2)
    s_grad = jax.lax.select(posdef, sg.copy(), scaling.dot(jnp.sign(sg) + (sg == 0)))
    s_newt = normalize(s_newt)
    s_grad = s_grad - s_newt * s_newt.dot(s_grad)
    subspace_other = jax.lax.select(
        jnp.linalg.norm(s_grad) > np_eps,
        jnp.vstack([s_newt, normalize(s_grad)]).T,
        jnp.vstack([s_newt, jnp.zeros(s_newt.shape)]).T
    )


    case0_cond = jnp.logical_and(posdef, jnp.linalg.norm(og_s_newt) < delta)
    subspace = jax.lax.select(case0_cond, subspace_0, subspace_other)
    
    s, ss, sc, og_s, og_ss, og_sc, qpval, br, iminbr, alpha = step_compute(
        x, subspace, sg, shess, delta, lb, ub, scaling, ss0, theta
    )

    ### TRT step ###
    trt_s0 = s0.at[iminbr].set(s0[iminbr] + theta * br[iminbr] * og_s[iminbr])
    trt_ss0 = ss0.at[iminbr].set(ss0[iminbr] + theta * br[iminbr] * og_ss[iminbr])
    # update x and at breakpoint
    trt_x = x + trt_s0

    trt_subspace = subspace.at[iminbr, :].set(0)
    # normalize subspace
    for ix in range(trt_subspace.shape[1]):
        # column normalization
        trt_subspace.at[:, ix].set(normalize(trt_subspace[:, ix]))

    # trt_s, trt_ss, trt_sc, trt_og_s, trt_og_ss, trt_og_sc, trt_qpval, _, _, _ = step_compute(
    #     trt_x, trt_subspace, sg, shess, delta, lb, ub, scaling, trt_ss0, theta
    # )

    trt_s, trt_ss, trt_sc, trt_og_s, trt_og_ss, trt_og_sc, trt_qpval, _, _, _ = jax.lax.cond(
        alpha < 1.0,
        step_compute,
        lambda *_: (s, ss, sc, og_s, og_ss, og_sc, jnp.inf, br, iminbr, alpha),
        trt_x, trt_subspace, sg, shess, delta, lb, ub, scaling, trt_ss0, theta
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
    def __init__(self, obj_fn, lb = None, ub = None, **kwargs):
        init_kwargs = kwargs
        if kwargs.get('options'):
            init_kwargs.update(kwargs['options'])
        if not lb is None:
            init_kwargs['lb'] = lb
        if not ub is None:
            init_kwargs['ub'] = ub
        self.init_kwargs = init_kwargs
        self.obj_fn = jax.jit(obj_fn)
        self.eps = np_eps
        self.state = None
        def update(state):
            step = tr_iteration(
                state['x'],
                state['grad'],
                state['hess'],
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
            loss, grad, hess = obj_fn(state['x_sol'])
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

            state['f_old'], state['x'], state['fval'], state['grad'], state['hess'], state['gnorm'] = jax_multi_where(
                state['accepted'],
                (state['fval'], state['x_sol'], loss, grad, hess, jnp.linalg.norm(state['grad'])),
                (state['f_old'], state['x'], state['fval'], state['grad'], state['hess'], state['gnorm'])
            )

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
            state['hess'],
        )

    # TODO: use StepInfo object for state
    def init_state(self, params, **kwargs):
        loss, grad, hess = self.obj_fn(params)
        return {
            'x': params,
            'fval': loss,
            'grad': grad,
            'gnorm': jnp.linalg.norm(grad),
            'hess': hess,
            'iter': 0,
            # optimizer params
            'lb': kwargs.get('lb') if kwargs.get('lb') is not None else -jnp.inf*jnp.ones(params.shape),
            'ub': kwargs.get('ub') if kwargs.get('ub') is not None else jnp.inf*jnp.ones(params.shape),
            'maxiter': kwargs.get('maxiter') or 1e3,
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
            'type': jnp.nan,
            's': jnp.nan,
            's0': jnp.nan,
            'ss': jnp.nan,
            'ss0': jnp.nan,
            'stepsx': jnp.nan,
            'nsx': jnp.nan,
            'normdx': jnp.nan,
            'accepted': False,
        }

    def converge_cond(self, state):
        return (
            state['tr_ratio'] > state['mu'] and state['f_diff'] < state['fatol'] + state['frtol'] * state['f_old']
            or state['iter'] > 1 and state['nsx'] < state['xtol']
            or state['gnorm'] <= state['gatol']
            or state['gnorm'] <= state['grtol'] * jnp.abs(state['f_old'])
        ) or (
            state['iter'] >= state['maxiter']
            or state['delta'] <= self.eps
        )

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

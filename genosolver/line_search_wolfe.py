import numpy as np
from scipy.optimize import _minpack2 as minpack2

from warnings import warn
class LineSearchWarning(UserWarning):
    pass

def line_search_wolfe1(f, fprime, xk, pk, gfk=None,
                       old_fval=None, old_old_fval=None,
                       args=(), c1=1e-4, c2=0.9, amax=50, amin=1e-8,
                       xtol=1e-14):
    """
    As `scalar_search_wolfe1` but do a line search to direction `pk`
    Parameters
    ----------
    f : callable
        Function `f(x)`
    fprime : callable
        Gradient of `f`
    xk : array_like
        Current point
    pk : array_like
        Search direction
    gfk : array_like, optional
        Gradient of `f` at point `xk`
    old_fval : float, optional
        Value of `f` at point `xk`
    old_old_fval : float, optional
        Value of `f` at point preceding `xk`
    The rest of the parameters are the same as for `scalar_search_wolfe1`.
    Returns
    -------
    stp, f_count, g_count, fval, old_fval
        As in `line_search_wolfe1`
    gval : array
        Gradient of `f` at the final point
    """
    if gfk is None:
        gfk = fprime(xk, *args)

    gval = [gfk]
    gc = [0]
    fc = [0]

    def phi(s):
        fc[0] += 1
        return f(xk + s*pk, *args)

    def derphi(s):
        gval[0] = fprime(xk + s*pk, *args)
        gc[0] += 1
        return np.dot(gval[0], pk)

    derphi0 = np.dot(gfk, pk)

    stp, fval, old_fval = scalar_search_wolfe1(
            phi, derphi, old_fval, old_old_fval, derphi0,
            c1=c1, c2=c2, amax=amax, amin=amin, xtol=xtol)

    return stp, fc[0], gc[0], fval, old_fval, gval[0]


def scalar_search_wolfe1(phi, derphi, phi0=None, old_phi0=None, derphi0=None,
                         c1=1e-4, c2=0.9,
                         amax=50, amin=1e-8, xtol=1e-14):
    """
    Scalar function search for alpha that satisfies strong Wolfe conditions
    alpha > 0 is assumed to be a descent direction.
    Parameters
    ----------
    phi : callable phi(alpha)
        Function at point `alpha`
    derphi : callable phi'(alpha)
        Objective function derivative. Returns a scalar.
    phi0 : float, optional
        Value of phi at 0
    old_phi0 : float, optional
        Value of phi at previous point
    derphi0 : float, optional
        Value derphi at 0
    c1 : float, optional
        Parameter for Armijo condition rule.
    c2 : float, optional
        Parameter for curvature condition rule.
    amax, amin : float, optional
        Maximum and minimum step size
    xtol : float, optional
        Relative tolerance for an acceptable step.
    Returns
    -------
    alpha : float
        Step size, or None if no suitable step was found
    phi : float
        Value of `phi` at the new point `alpha`
    phi0 : float
        Value of `phi` at `alpha=0`
    Notes
    -----
    Uses routine DCSRCH from MINPACK.
    """

    if phi0 is None:
        phi0 = phi(0.)
    if derphi0 is None:
        derphi0 = derphi(0.)

    if old_phi0 is not None and derphi0 != 0:
        alpha1 = min(1.0, 1.01*2*(phi0 - old_phi0)/derphi0)
        if alpha1 < 0:
            alpha1 = 1.0
    else:
        alpha1 = 1.0
        
    # need to have alpha1 within the corresponding bounds
    alpha1 = max(min(alpha1, amax), amin)

    phi1 = phi0
    derphi1 = derphi0
    isave = np.zeros((2,), np.intc)
    dsave = np.zeros((13,), float)
    task = b'START'

    maxiter = 100
    for i in range(maxiter):
        stp, phi1, derphi1, task = minpack2.dcsrch(alpha1, phi1, derphi1,
                                                   c1, c2, xtol, task,
                                                   amin, amax, isave, dsave)
        if task[:2] == b'FG':
            alpha1 = stp
            phi1 = phi(stp)
            derphi1 = derphi(stp)
        else:
            break

    if task[:5] == b'ERROR':
        stp = None  # failed
    if task[:7] == b'WARNING':
        print('WARNING')

    return stp, phi1, phi0

def line_search_wolfe2(f, fprime, xk, pk, gfk=None, old_fval=None,
                       old_old_fval=None, args=(), c1=1e-4, c2=0.9, amax=None,
                       extra_condition=None, maxiter=10):
    """Find alpha that satisfies strong Wolfe conditions.

    Parameters
    ----------
    f : callable f(x,*args)
        Objective function.
    fprime : callable f'(x,*args)
        Objective function gradient.
    xk : ndarray
        Starting point.
    pk : ndarray
        Search direction.
    gfk : ndarray, optional
        Gradient value for x=xk (xk being the current parameter
        estimate). Will be recomputed if omitted.
    old_fval : float, optional
        Function value for x=xk. Will be recomputed if omitted.
    old_old_fval : float, optional
        Function value for the point preceding x=xk.
    args : tuple, optional
        Additional arguments passed to objective function.
    c1 : float, optional
        Parameter for Armijo condition rule.
    c2 : float, optional
        Parameter for curvature condition rule.
    amax : float, optional
        Maximum step size
    extra_condition : callable, optional
        A callable of the form ``extra_condition(alpha, x, f, g)``
        returning a boolean. Arguments are the proposed step ``alpha``
        and the corresponding ``x``, ``f`` and ``g`` values. The line search
        accepts the value of ``alpha`` only if this
        callable returns ``True``. If the callable returns ``False``
        for the step length, the algorithm will continue with
        new iterates. The callable is only called for iterates
        satisfying the strong Wolfe conditions.
    maxiter : int, optional
        Maximum number of iterations to perform.

    Returns
    -------
    alpha : float or None
        Alpha for which ``x_new = x0 + alpha * pk``,
        or None if the line search algorithm did not converge.
    fc : int
        Number of function evaluations made.
    gc : int
        Number of gradient evaluations made.
    new_fval : float or None
        New function value ``f(x_new)=f(x0+alpha*pk)``,
        or None if the line search algorithm did not converge.
    old_fval : float
        Old function value ``f(x0)``.
    new_slope : float or None
        The local slope along the search direction at the
        new value ``<fprime(x_new), pk>``,
        or None if the line search algorithm did not converge.


    Notes
    -----
    Uses the line search algorithm to enforce strong Wolfe
    conditions. See Wright and Nocedal, 'Numerical Optimization',
    1999, pp. 59-61.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.optimize import line_search

    A objective function and its gradient are defined.

    >>> def obj_func(x):
    ...     return (x[0])**2+(x[1])**2
    >>> def obj_grad(x):
    ...     return [2*x[0], 2*x[1]]

    We can find alpha that satisfies strong Wolfe conditions.

    >>> start_point = np.array([1.8, 1.7])
    >>> search_gradient = np.array([-1.0, -1.0])
    >>> line_search(obj_func, obj_grad, start_point, search_gradient)
    (1.0, 2, 1, 1.1300000000000001, 6.13, [1.6, 1.4])

    """
    fc = [0]
    gc = [0]
    gval = [None]
    gval_alpha = [None]

    def phi(alpha):
        fc[0] += 1
        return f(xk + alpha * pk, *args)

    fprime = fprime

    def derphi(alpha):
        gc[0] += 1
        gval[0] = fprime(xk + alpha * pk, *args)  # store for later use
        gval_alpha[0] = alpha
        return np.dot(gval[0], pk)

    if gfk is None:
        gfk = fprime(xk, *args)
    derphi0 = np.dot(gfk, pk)

    if extra_condition is not None:
        # Add the current gradient as argument, to avoid needless
        # re-evaluation
        def extra_condition2(alpha, phi):
            if gval_alpha[0] != alpha:
                derphi(alpha)
            x = xk + alpha * pk
            return extra_condition(alpha, x, phi, gval[0])
    else:
        extra_condition2 = None

    alpha_star, phi_star, old_fval, derphi_star = scalar_search_wolfe2(
            phi, derphi, old_fval, old_old_fval, derphi0, c1, c2, amax,
            extra_condition2, maxiter=maxiter)

    if derphi_star is None:
        warn('The line search algorithm did not converge', LineSearchWarning)
    else:
        # derphi_star is a number (derphi) -- so use the most recently
        # calculated gradient used in computing it derphi = gfk*pk
        # this is the gradient at the next step no need to compute it
        # again in the outer loop.
        derphi_star = gval[0]

    return alpha_star, fc[0], gc[0], phi_star, old_fval, derphi_star


def scalar_search_wolfe2(phi, derphi, phi0=None,
                         old_phi0=None, derphi0=None,
                         c1=1e-4, c2=0.9, amax=None,
                         extra_condition=None, maxiter=10):
    """Find alpha that satisfies strong Wolfe conditions.

    alpha > 0 is assumed to be a descent direction.

    Parameters
    ----------
    phi : callable phi(alpha)
        Objective scalar function.
    derphi : callable phi'(alpha)
        Objective function derivative. Returns a scalar.
    phi0 : float, optional
        Value of phi at 0.
    old_phi0 : float, optional
        Value of phi at previous point.
    derphi0 : float, optional
        Value of derphi at 0
    c1 : float, optional
        Parameter for Armijo condition rule.
    c2 : float, optional
        Parameter for curvature condition rule.
    amax : float, optional
        Maximum step size.
    extra_condition : callable, optional
        A callable of the form ``extra_condition(alpha, phi_value)``
        returning a boolean. The line search accepts the value
        of ``alpha`` only if this callable returns ``True``.
        If the callable returns ``False`` for the step length,
        the algorithm will continue with new iterates.
        The callable is only called for iterates satisfying
        the strong Wolfe conditions.
    maxiter : int, optional
        Maximum number of iterations to perform.

    Returns
    -------
    alpha_star : float or None
        Best alpha, or None if the line search algorithm did not converge.
    phi_star : float
        phi at alpha_star.
    phi0 : float
        phi at 0.
    derphi_star : float or None
        derphi at alpha_star, or None if the line search algorithm
        did not converge.

    Notes
    -----
    Uses the line search algorithm to enforce strong Wolfe
    conditions. See Wright and Nocedal, 'Numerical Optimization',
    1999, pp. 59-61.

    """

    if phi0 is None:
        phi0 = phi(0.)

    if derphi0 is None:
        derphi0 = derphi(0.)

    alpha0 = 0
    if old_phi0 is not None and derphi0 != 0:
        alpha1 = min(1.0, 1.01*2*(phi0 - old_phi0)/derphi0)
    else:
        alpha1 = 1.0

    if alpha1 < 0:
        alpha1 = 1.0

    if amax is not None:
        alpha1 = min(alpha1, amax)

    phi_a1 = phi(alpha1)
    #derphi_a1 = derphi(alpha1) evaluated below

    phi_a0 = phi0
    derphi_a0 = derphi0

    if extra_condition is None:
        extra_condition = lambda alpha, phi: True

    for i in range(maxiter):
        if alpha1 == 0 or (amax is not None and alpha0 == amax):
            # alpha1 == 0: This shouldn't happen. Perhaps the increment has
            # slipped below machine precision?
            phi_star = phi0
            phi0 = old_phi0

            if alpha1 == 0:
                alpha_star = None
                derphi_star = None
                msg = 'Rounding errors prevent the line search from converging'
            else:
                alpha_star = alpha0
                derphi_star = derphi_a0
                msg = "The line search algorithm could not find a solution " + \
                      "less than or equal to amax: %s" % amax

            warn(msg, LineSearchWarning)
            break

        not_first_iteration = i > 0
        if (phi_a1 > phi0 + c1 * alpha1 * derphi0) or \
           ((phi_a1 >= phi_a0) and not_first_iteration):
            alpha_star, phi_star, derphi_star = \
                        _zoom(alpha0, alpha1, phi_a0,
                              phi_a1, derphi_a0, phi, derphi,
                              phi0, derphi0, c1, c2, extra_condition)
            break

        derphi_a1 = derphi(alpha1)
        if (abs(derphi_a1) <= -c2*derphi0):
            if extra_condition(alpha1, phi_a1):
                alpha_star = alpha1
                phi_star = phi_a1
                derphi_star = derphi_a1
                break

        if (derphi_a1 >= 0):
            alpha_star, phi_star, derphi_star = \
                        _zoom(alpha1, alpha0, phi_a1,
                              phi_a0, derphi_a1, phi, derphi,
                              phi0, derphi0, c1, c2, extra_condition)
            break

        alpha2 = 4 * alpha1  # increase by factor of **two** on each iteration
        if amax is not None:
            alpha2 = min(alpha2, amax)
        alpha0 = alpha1
        alpha1 = alpha2
        phi_a0 = phi_a1
        phi_a1 = phi(alpha1)
        derphi_a0 = derphi_a1

    else:
        # stopping test maxiter reached
        alpha_star = alpha1
        phi_star = phi_a1
        derphi_star = derphi_a1
        warn('The line search algorithm did not converge', LineSearchWarning)

    return alpha_star, phi_star, phi0, derphi_star


def _cubicmin(a, fa, fpa, b, fb, c, fc):
    """
    Finds the minimizer for a cubic polynomial that goes through the
    points (a,fa), (b,fb), and (c,fc) with derivative at a of fpa.

    If no minimizer can be found, return None.

    """
    # f(x) = A *(x-a)^3 + B*(x-a)^2 + C*(x-a) + D

    with np.errstate(divide='raise', over='raise', invalid='raise'):
        try:
            C = fpa
            db = b - a
            dc = c - a
            denom = (db * dc) ** 2 * (db - dc)
            d1 = np.empty((2, 2))
            d1[0, 0] = dc ** 2
            d1[0, 1] = -db ** 2
            d1[1, 0] = -dc ** 3
            d1[1, 1] = db ** 3
            [A, B] = np.dot(d1, np.asarray([fb - fa - C * db,
                                            fc - fa - C * dc]).flatten())
            A /= denom
            B /= denom
            radical = B * B - 3 * A * C
            xmin = a + (-B + np.sqrt(radical)) / (3 * A)
        except ArithmeticError:
            return None
    if not np.isfinite(xmin):
        return None
    return xmin


def _quadmin(a, fa, fpa, b, fb):
    """
    Finds the minimizer for a quadratic polynomial that goes through
    the points (a,fa), (b,fb) with derivative at a of fpa.

    """
    # f(x) = B*(x-a)^2 + C*(x-a) + D
    with np.errstate(divide='raise', over='raise', invalid='raise'):
        try:
            D = fa
            C = fpa
            db = b - a * 1.0
            B = (fb - D - C * db) / (db * db)
            xmin = a - C / (2.0 * B)
        except ArithmeticError:
            return None
    if not np.isfinite(xmin):
        return None
    return xmin


def _zoom(a_lo, a_hi, phi_lo, phi_hi, derphi_lo,
          phi, derphi, phi0, derphi0, c1, c2, extra_condition):
    """Zoom stage of approximate linesearch satisfying strong Wolfe conditions.
    
    Part of the optimization algorithm in `scalar_search_wolfe2`.
    
    Notes
    -----
    Implements Algorithm 3.6 (zoom) in Wright and Nocedal,
    'Numerical Optimization', 1999, pp. 61.

    """

    maxiter = 10
    i = 0
    delta1 = 0.2  # cubic interpolant check
    delta2 = 0.1  # quadratic interpolant check
    phi_rec = phi0
    a_rec = 0
    while True:
        # interpolate to find a trial step length between a_lo and
        # a_hi Need to choose interpolation here. Use cubic
        # interpolation and then if the result is within delta *
        # dalpha or outside of the interval bounded by a_lo or a_hi
        # then use quadratic interpolation, if the result is still too
        # close, then use bisection

        dalpha = a_hi - a_lo
        if dalpha < 0:
            a, b = a_hi, a_lo
        else:
            a, b = a_lo, a_hi

        # minimizer of cubic interpolant
        # (uses phi_lo, derphi_lo, phi_hi, and the most recent value of phi)
        #
        # if the result is too close to the end points (or out of the
        # interval), then use quadratic interpolation with phi_lo,
        # derphi_lo and phi_hi if the result is still too close to the
        # end points (or out of the interval) then use bisection

        if (i > 0):
            cchk = delta1 * dalpha
            a_j = _cubicmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi,
                            a_rec, phi_rec)
        if (i == 0) or (a_j is None) or (a_j > b - cchk) or (a_j < a + cchk):
            qchk = delta2 * dalpha
            a_j = _quadmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi)
            if (a_j is None) or (a_j > b-qchk) or (a_j < a+qchk):
                a_j = a_lo + 0.5*dalpha

        # Check new value of a_j

        phi_aj = phi(a_j)
        if (phi_aj > phi0 + c1*a_j*derphi0) or (phi_aj >= phi_lo):
            phi_rec = phi_hi
            a_rec = a_hi
            a_hi = a_j
            phi_hi = phi_aj
        else:
            derphi_aj = derphi(a_j)
            if abs(derphi_aj) <= -c2*derphi0 and extra_condition(a_j, phi_aj):
                a_star = a_j
                val_star = phi_aj
                valprime_star = derphi_aj
                break
            if derphi_aj*(a_hi - a_lo) >= 0:
                phi_rec = phi_hi
                a_rec = a_hi
                a_hi = a_lo
                phi_hi = phi_lo
            else:
                phi_rec = phi_lo
                a_rec = a_lo
            a_lo = a_j
            phi_lo = phi_aj
            derphi_lo = derphi_aj
        i += 1
        if (i > maxiter):
            # Failed to find a conforming step size
            a_star = None
            val_star = None
            valprime_star = None
            break
    return a_star, val_star, valprime_star

from linesearch_allnew import dcsrch

def line_search_wolfe3(f, fprime, xk, pk, gfk=None,
                       old_fval=None, old_old_fval=None,
                       args=(), c1=1e-4, c2=0.9, amax=50., amin=1e-14,
                       xtol=0.1, verbose=0):

    stp = 1.
    stp = max(amin, stp)
    stp = min(stp, amax)
    
    task = 'START'
    ftol = c1
    gtol = c2

    def phi(s):
        fx = f(xk + s*pk)
        gx = fprime(xk + s*pk)
        gd = np.dot(gx, pk)
        return fx, gd, gx
    
    cnt_on = 0
    if old_fval is None or gfk is None:
        old_fval, _, gfk = phi(0)
        cnt_on = 1

    fval = old_fval
    gval = np.dot(gfk, pk)
    fval, grad, stp, task, fg_cnt = dcsrch(fval, gval, stp, ftol, gtol, xtol, amin, amax, task, phi, verbose,grad=gfk)

    if old_fval is None or gfk is None:
        fg_cnt += 1
    
    if task == 'ERROR':
        print('Error in line search')
        raise Exception('Line search error')

    return stp, (fg_cnt+cnt_on), 0, fval, old_fval, grad


def line_search_wolfe3_debug(f, fprime, xk, pk, gfk=None,
                             old_fval=None, old_old_fval=None,
                             args=(), c1=1e-4, c2=0.9, amax=50., amin=1e-14,
                             xtol=0.1, verbose=10, plot_path='./steps'):

    stp = 1.
    stp = max(amin, stp)
    stp = min(stp, amax)
    
    task = 'START'
    ftol = c1
    gtol = c2

    steps_array = []
    def phi(s):
        steps_array.append(s)
        fx = f(xk + s*pk)
        gx = fprime(xk + s*pk)
        gd = np.dot(gx, pk)
        return fx, gd, gx
    
    cnt_on = 0
    if old_fval is None or gfk is None:
        old_fval, _, gfk = phi(0)
        cnt_on = 1

    fval = old_fval
    gval = np.dot(gfk, pk)
    fval, grad, stp, task, fg_cnt = dcsrch(fval, gval, stp, ftol, gtol, xtol, amin, amax, task, phi, verbose,grad=gfk)

    if old_fval is None or gfk is None:
        fg_cnt += 1

    steps_array.append(stp)
    import matplotlib.pyplot as plt
    import os, datetime
    os.makedirs(plot_path, exist_ok=True)
    fig, axs = plt.subplots(nrows=1, ncols=2)
    axs[0].plot(list(range(len(steps_array))), steps_array)
    axs[1].semilogy(list(range(len(steps_array))), steps_array)
    plot_name = datetime.datetime.now().strftime('%Y-%m-%d-%H_%M_%S')
    plt.save(os.path.join(plot_path, f'{plot_name}.pdf'))
    plt.close()
    
    if task == 'ERROR':
        print('Error in line search')
        raise Exception('Line search error')

    return stp, (fg_cnt+cnt_on), 0, fval, old_fval, grad

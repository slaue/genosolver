from typing import Callable, Optional
import numpy as np


def zipNaN(phi: Callable[[float],tuple[float,np.ndarray]],
           alpha: float,
           delta: float,
           stp: float=1.
           )-> tuple[float, np.ndarray, float, int]:

    fg_cnt = 0
    for _i in range(20):
        f, g = phi(delta + stp*alpha)
        fg_cnt += 1
        if np.isneginf(f) or np.isfinite(f) and np.isfinite(g).all():
            return f, g, alpha, fg_cnt
        
        alpha = .5 * alpha

    return f, g, None, fg_cnt

from numpy.polynomial import Polynomial as Poly

def pfval(r, t, f0, f1, gd0, gd1, c):
    return t**(3/2)*np.sqrt((r*(1-r))**3)/(t*r*(1-r)*((1-r)*gd0-r*gd1)+(2*r**3-3*r**2+1)*f0+(3*r**2-2*r**3)*f1-c)

def pval(t, f0, f1, gd0, gd1, c):
    r = Poly([ 3*(f0-c), 6*(c-f0) + t*gd0, t*(gd1 - 2*gd0) + 3*(f0-f1), t*(gd0-gd1) ]).roots()
    r = r[(r > 0) & (r < 1) & np.isreal(r)]
    r = np.array(r, dtype=float)
    vals = t**(3/2)*np.sqrt((r*(1-r))**3)/(t*r*(1-r)*((1-r)*gd0-r*gd1)+(2*r**3-3*r**2+1)*f0+(3*r**2-2*r**3)*f1-c)
    indx = np.argmax(vals)
    return r[indx], vals[indx]



def line_search_wolfe45(fg, xk, d, g=None,
                        old_fval=None, old_old_fval=None,
                        args=(), c1=1e-4, c2=0.9, amax=50., amin=1e-14,
                        xtol=1e-14, verbose=0, np=np):

    stp = np.clip(1., amin, amax)

    def phi(s):
        fx, gx = fg(xk + s*d)
        return fx, gx

    fg_cnt = 0
    if old_fval is None or g is None:
        old_fval, g = phi(0)
        fg_cnt += 1

    delta = 0.
    alpha = 1.
    eps = 0.0
    finit = old_fval
    gdinit = g.dot(d)
    gtest = c1*gdinit
    g_old = g
    f_low = finit
    gd_low = gdinit

    for _j in range(20):
        f, g, alpha, fg_new = zipNaN(phi, alpha, delta, stp)
        fg_cnt += fg_new
        if alpha is None:
            print('No step size found')
            return None, fg_cnt, finit, g_old
        if np.isneginf(f):
            return delta + stp*alpha, fg_cnt, f, g
        gd = g.dot(d)

        if f >= f_low or gd >= c2*gdinit:
            break
        gd_low = gd
        f_low = f
        delta = delta + alpha
        alpha = 4. * alpha
    else:
        return (delta-alpha/4.) + alpha/4.*stp, fg_cnt, f, g


    stp = delta + alpha*stp

    segs = np.array([delta,stp])
    fvals = np.array([f_low, f])
    gvals = np.array([g_low, g])

    best_f = f
    best_g = np.dot(g,d)
    best_stp = stp
    
    for i in range(1,20):
        ftest = finit + stp*gtest
        if f < ftest and abs(np.dot(g,d)) <= c2*(-gdinit):
            if verbose >= 99:
                print('STRONG WOLFE SATISFIED')
            best_f = f
            best_g = g
            best_stp = stp
            break

        nxtstp = None
        nxtp = float('-inf')
        for j in range(i):
            nes, nep = pval(segs[j+1] - segs[j], fvals[j], fvals[j+1], gvals[j], gvals[j+1], fvals.min() - 1./i)
            if nep > nxtp:
                nxtstp = nes*(segs[j+1] - segs[j]) + segs[j]
                nxtp = nep

        f, g, stp_new, fg_new = zipNaN(phi, nxtstp-delta, delta)
        fg_cnt += fg_new
        stp_new += delta
        if stp_new != nxtstp:
            indx = (segs < stp_new)
        
        stp = stp_new
    
        if f < best_f:
            best_f = f
            best_g = g
            best_stp = stp
        
        segs = np.array(list(segs) + [stp])
        fvals = np.array(list(fvals) + [f])
        gvals = np.array(list(gvals) + [np.dot(g,d)])

        indx = np.argsort(segs)
        segs = segs[indx]
        fvals = fvals[indx]
        gvals = gvals[indx]
    
    return best_stp, fg_cnt, best_f, best_g

if __name__ == '__main__':
    pass

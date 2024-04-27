from typing import Callable, Optional
from queue import PriorityQeueu
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

def pval(t, f0, f1, gd0, gd1, c):
    r = np.polynomial.Poly([ 3*(f0-c), 6*(c-f0) + t*gd0, t*(gd1 - 2*gd0) + 3*(f0-f1), t*(gd0-gd1) ]).roots()
    r = roots[(r> 0) & (r < 1) & np.isreal(r)]
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
    
    Q = PriorityQueue()

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

    #####
    cub = Cubic(delta, (delta + alpha*stp), f_low, gd_low, f, gd, alpha=(1.+abs(f-finit))*stp, gamma=.5)
    xm, fxm = cub.min
    Q.put((fxm, -xm, cub))

    stp = delta + alpha*stp
    best_f = f
    best_g = g
    best_stp = stp
    
    for _i in range(20):
        
        ftest = finit + stp*gtest
        if f < ftest and abs(g.dot(d)) <= c2 * (-gdinit):
            if verbose >= 99:
                print('STRONG WOLFE SATISFIED')
            best_f = f
            best_g = g
            best_stp = stp
            
            break

        if Q.empty():
            if verbose >= 99:
                print('No reasonable stepsize found')
            break
        
        cub = Q.get()[-1]
        x0 = cub.x0
        x1 = cub.x1
        if (x1 - x0) < xtol * x1:
            if verbose >= 99:
                print('XTOL line search')
                print('Skipped', x0, cub.min[0], x1)
            continue

        stp = cub.min[0]
        if not (x0 + (x1 - x0) * .1 <= stp <= x1 - (x1 - x0) * .1):
            stp = np.clip(stp, x0 + (x1 - x0) * .1, x1 - (x1 - x0) * .1)
            if verbose >= 99:
                print('step on boundary, clipped')
        f, g = phi(stp)
        fg_cnt += 1

        for _j in range(20):
            if np.isneginf(f):
                break
            if np.isfinite(f) and np.isfinite(g).all():
                break

            if verbose >= 99:
                print('f or g has inf or nan')

            stp = .5 * stp
            x1 = stp
            f, g = phi(stp)
            fg_cnt += 1
        else:
            continue
        
        if (f, abs(g.dot(d)), -stp) < (best_f, abs(best_g.dot(d)), -best_stp):
            best_f = f
            best_g = g
            best_stp = stp
        
        cubl = Cubic(x0, stp, cub.f(x0), cub.g(x0), f, g.dot(d), alpha=(1.+abs(f-cub.f(x0)))*(stp-x0), gamma=.5)
        xl, fxl = cubl.min
        Q.put((fxl, -xl, cubl))

        if stp < x1:
            cubr = Cubic(stp, x1, f, g.dot(d), cub.f(x1), cub.g(x1), alpha=(1.+abs(f-cub.f(x1)))*(x1-stp), gamma=.5)
            xr, fxr = cubr.min
            Q.put((fxr, -xr, cubr))
    else:
        if verbose >= 99:
            print('MAX ITER line search')

    #if best_stp > amin:
    #    best_stp = amin
    #    fg_cnt += 1
    #    best_f, best_g = phi(amin)
    
    return best_stp, fg_cnt, best_f, best_g


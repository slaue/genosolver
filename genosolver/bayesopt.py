from abc import ABC, abstractmethod
from typing import Callable, Optional
from functools import wraps
import matplotlib.pyplot as plt
from autograd import grad, elementwise_grad, jacobian, value_and_grad
import autograd.numpy as np
import autograd.scipy.stats as stats
import autograd.scipy.linalg as lin
import warnings

class Optimizable(ABC):

    __slots__ = [ 'theta', 'bounds' ]

    theta: np.ndarray
    bounds: list[tuple[float,float]]

    @property
    def parameters(self)-> np.ndarray:
        return self.theta

    @parameters.setter
    def parameters(self, x: np.ndarray):
        self.theta = x

    @abstractmethod
    def __call__(self, x: np.ndarray)-> np.ndarray:
        ...

    @abstractmethod
    def grad(self, x: np.ndarray)-> np.ndarray:
        ...

def mu_grad(mu: Callable[[np.ndarray],np.ndarray])-> Callable[[np.ndarray],np.ndarray]:
    @wraps(mu)
    def fn_mu(x: np.ndarray)-> np.ndarray:
        A = mu(x)
        B = elementwise_grad(mu)(x)
        return np.concatenate([A,B])
    return fn_mu

def ker_grad(ker: Callable[[np.ndarray],np.ndarray])-> Callable[[np.ndarray],np.ndarray]:
    @wraps(ker)
    def fn_ker(x: np.ndarray)-> np.ndarray:
        A = ker(x)
        g0 = elementwise_grad(ker)
        C = g0(x) # Not working?
        B = -C
        g01 = elementwise_grad(g0)
        D = -g01(x)
        E1 = np.concatenate([A, B],axis=1)
        E2 = np.concatenate([C, D],axis=1)
        E = np.concatenate([E1,E2],axis=0)
        return E
    return fn_ker

class Kernel(Optimizable):

    def grad(self, x: np.ndarray)-> np.ndarray:
        return ker_grad(self)(x)

class Expectation(Optimizable):

    def grad(self, x: np.ndarray)-> np.ndarray:
        return mu_grad(self)(x)

class Matern52(Kernel):
    '''
    Matern's 5/2 Kernel
    WARNING: autograd cannot calculate d^2/(dx)^2 (solution is a hack)
    '''

    def __init__(self, sig: float=1.5, theta: float=.1, bounds: list[tuple[float,float]]=((1e-18,np.inf),(1e-10,1e5))):
        self.theta = np.array([ sig, theta ])
        self.bounds = bounds

    def __call__(self, d: np.ndarray)-> np.ndarray:
        xdr = self.theta[1]*np.sqrt(5)*abs(d)
        return self.theta[0]*(1 + xdr + xdr**2/3)*np.exp(-xdr)

    def grad(self, x: np.ndarray)-> np.ndarray:
        y = super().grad(x)
        c = self.theta[0]*self.theta[1]**2*5/3
        mask = np.block([ [np.zeros_like(x), np.zeros_like(x)], [np.zeros_like(x), x==0] ])
        y = y + c * mask
        return y

class RBF(Kernel):
    '''
    Gaussian Kernel
    '''

    def __init__(self, alpha: float=10., theta: float=.1, bounds: list[tuple[float,float]]=((1e-10,1e5),(1e-10,1e5))):
        self.theta = np.array([ alpha, theta ])
        self.bounds = bounds

    def __call__(self, d: np.ndarray)-> np.ndarray:
        return self.theta[0] * np.exp(-self.theta[1]/2.*(d)**2)
    
    def grad(self, d: np.ndarray)-> np.ndarray:
        A = self(d)
        B = -self.theta[1]*d*A
        C = -B
        D = -self.theta[1]*d*C + self.theta[1]*A
        E1 = np.concatenate([A, C],axis=1)
        E2 = np.concatenate([B, D], axis=1)
        E = np.concatenate([E1, E2],axis=0)
        return E
    
class RQK(Kernel):
    '''
    Rational Quadratic Kernel
    '''

    def __init__(self, alpha: float=10., theta: float=1., p: float=1., bounds: list[tuple[float,float]]=((1e-18,np.inf),(1e-10,1e5),(1e-5,1e3))):
        self.theta = np.array([ alpha, theta, p ])
        self.bounds = bounds

    def __call__(self, d: np.ndarray)-> np.ndarray:
        return self.theta[0]*(1+self.theta[1]*d**2*self.theta[2])**(-self.theta[2])

class CubicSpline(Kernel):
    '''
    SPH Cubic Spline Kernel
    '''

    def __init__(self, alpha: float=10., delta: float=1., bounds: list[tuple[float,float]]=((1e-10,1e10),(1e-10,2.))):
        self.theta = np.array([ alpha, delta ])
        self.bounds = bounds

    def __call__(self, d: np.ndarray)-> np.ndarray:
        mask0 = np.zeros_like(d)
        mask1 = np.zeros_like(d)
        mask2 = np.zeros_like(d)
        da = self.theta[1]*np.abs(d)
        mask1[(da<=1)] = 1.
        mask2[(da<=2) & (da>1)] = 1.
        E = mask1*(1-3/2*da**2*(1-da/2))
        E = E + mask2*(2-da)**3*4
        return E
    
    def grad(self, d: np.ndarray)-> np.ndarray:
        mask1 = np.zeros_like(d)
        mask2 = np.zeros_like(d)
        da = self.theta[1]*np.abs(d)
        mask1[da<1] = 1.
        mask2[da<2] = 1.
        mask2[mask1==1.] = 0.
        A = mask1*(1-3/2*da**2*(1-da/2)) + mask2*(2-da)**3/4
        B = mask1*(3/4*da*(3*da-4)) + mask2*(-3/4*(2-da)**2)
        B = self.theta[1]*np.sign(d)*B
        C = -B
        D = mask1*(9/2*da-3) + mask2*(3-3/2*da)
        D = -self.theta[1]**2*D
        E1 = np.concatenate([A, C],axis=1)
        E2 = np.concatenate([B, D], axis=1)
        E = np.concatenate([E1, E2],axis=0)
        E = self.theta[0]*E
        return E

    
def polyval(p, x):
    mx = np.zeros_like(x)
    for v in p:
        mx *= x
        mx += v
    return mx

class PolyRegressor(Expectation):
    '''
    Polynomial regression
    '''

    def __init__(self, p: np.ndarray, bounds: list[tuple[float,float]]=None):
        self.theta = p
        self.bounds = [(-np.inf,np.inf)]*len(p) if bounds is None else bounds

    def __call__(self, x: np.ndarray)-> np.ndarray:
        return polyval(self.theta, x)

class GaussianProcess:
    __slots__ = [ 'mu', 'ker', 'reg', 'x', 'y', 'g', '_L' ]

    mu: Callable[[np.ndarray], np.ndarray]
    ker: Callable[[np.ndarray, np.ndarray], np.ndarray]
    reg: float
    x: np.ndarray
    y: np.ndarray
    g: Optional[np.ndarray]
    _L: np.ndarray

    def __init__(self,
                 mu: Callable[[np.ndarray], np.ndarray],
                 ker: Callable[[np.ndarray], np.ndarray],
                 reg: float=None
        ):
        self.mu = mu
        self.ker = ker
        self.reg = reg
        self.x = np.array([])
        self.y = np.array([])
        self.g = None
        self._L = np.array([])

    def expect(self, x: np.ndarray)-> np.ndarray:
        x = np.atleast_1d(x)
        dff = x[:,None]-self.x
        cov = self.ker(dff) if self.g is None else self.ker.grad(dff)
        diff = (self.y - self.mu(self.x)) if self.g is None else (np.concatenate((self.y, self.g)) - self.mu.grad(self.x))
        v = lin.solve_triangular(self._L, diff, lower=True)
        wt = lin.solve_triangular(self._L, cov.T, lower=True)
        mux = self.mu(x) if self.g is None else self.mu.grad(x)
        return mux + wt.T @ v

    def covary(self, x: np.ndarray)-> np.ndarray:
        x = np.atleast_1d(x)
        dff = x[:,None]-self.x
        covL = self.ker(dff) if self.g is None else self.ker.grad(dff)
        covR = self.ker(-dff.T) if self.g is None else self.ker.grad(-dff.T)
        VR = lin.solve_triangular(self._L, covR, lower=True)
        VLt = lin.solve_triangular(self._L, covL.T, lower=True)
        kro = self.ker(x[:,None]-x) if self.g is None else self.ker.grad(x[:,None]-x)
        return kro - VLt.T @ VR

    def predict(self, x: np.ndarray)-> tuple[np.ndarray, np.ndarray]:
        x = np.atleast_1d(x)
        dff = x[:,None]-self.x
        covL = self.ker(dff) if self.g is None else self.ker.grad(dff)
        covR = self.ker(-dff.T) if self.g is None else self.ker.grad(-dff.T)
        diff = (self.y - self.mu(self.x)) if self.g is None else (np.concatenate((self.y, self.g)) - self.mu.grad(self.x))
        VR = lin.solve_triangular(self._L, covR, lower=True)
        VLt = lin.solve_triangular(self._L, covL.T, lower=True)
        w = lin.solve_triangular(self._L, diff, lower=True)
        mux = self.mu(x) if self.g is None else self.mu.grad(x)
        kro = self.ker(x[:,None]-x) if self.g is None else self.ker.grad(x[:,None]-x)
        return mux + VR.T @ w, kro - VLt.T @ VR

    def update(self, *,
               mu: Optional[Callable[[np.ndarray], np.ndarray]]=None,
               ker: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]]=None,
               reg: Optional[float]=None):
        if mu is not None: self.mu = mu
        if ker is not None: self.ker = ker
        if reg is not None: self.reg = reg
        kerg = self.ker if self.g is None else self.ker.grad
        if self.reg is not None:
            D = self.reg * np.eye(self.x.shape[0] if self.g is None else 2*self.x.shape[0])
            self._L = np.linalg.cholesky(kerg(self.x[:,None]-self.x) + D)
        else:
            self._L = np.linalg.cholesky(kerg(self.x[:,None]-self.x))

    def add(self, x: np.ndarray, y: np.ndarray, g: Optional[np.ndarray]=None):
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        self.x = np.concatenate([ self.x, x ])
        self.y = np.concatenate([ self.y, y ])
        if g is not None:
            g = np.atleast_1d(g)
            self.g = g if self.g is None else np.concatenate([ self.g, g ])
        self.update()

    def EI(self, x: np.ndarray)-> np.ndarray:
        x = np.atleast_1d(x)
        mu, sig = self.predict(x)
        sig = np.diag(sig)

        mn = self.y.min()
        #indx = (sig != 0)
        z = np.zeros(x.shape[0])
        h = np.zeros(x.shape[0])

        z = (mn - mu) / (sig + 1e-16)
        h = (mn - mu) * stats.norm.cdf(z) + sig * stats.norm.pdf(z)
        return np.maximum(h, 0.)

    def UCB(self, x: np.ndarray, beta: float=2.)-> np.ndarray:
        x = np.atleast_1d(x)
        mu, sig = self.predict(x)
        sig = np.diag(sig)
        return mu + beta*sig

    def logL(self)-> float:
        diff = (self.y - self.mu(self.x)) if self.g is None else (np.concatenate((self.y, self.g)) - self.mu.grad(self.x))
        z = lin.solve_triangular(self._L, diff, lower=True)
        return -np.sum(np.log(np.diag(self._L))) - .5 * np.dot(z.T, z) - .5 * self.x.shape[0]*np.log(2.*np.pi)

def plot_gp(gp: GaussianProcess, f=None, g=None):
    n = 100
    t = np.linspace(gp.x.min(), gp.x.max(), n)
    ex, cov = gp.predict(t)
    if gp.g is None:
        if f is not None:
            plt.plot(t, [f(s) for s in t], '-.')
        plt.plot(t, ex)
        plt.plot(t, ex + 2 * np.diagonal(cov), '--g')
        plt.plot(t, ex - 2 * np.diagonal(cov), '--r')
        plt.plot(gp.x, gp.y, 'x')
        plt.show()
    else:
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
        if f is not None:
            axs[0].plot(t, [f(s) for s in t], '-.')
        axs[0].plot(t, ex[:n])
        axs[0].plot(t, ex[:n] + 2 * np.diagonal(cov)[:n], '--g')
        axs[0].plot(t, ex[:n] - 2 * np.diagonal(cov)[:n], '--r')
        axs[0].plot(gp.x, gp.y, 'x')
        if g is not None:
            axs[1].plot(t, [g(s) for s in t], '-.')
        axs[1].plot(t, ex[n:])
        axs[1].plot(t, ex[n:] + 2 * np.diagonal(cov)[n:], '--g')
        axs[1].plot(t, ex[n:] - 2 * np.diagonal(cov)[n:], '--r')
        axs[1].plot(gp.x, gp.g, 'x')
        plt.show()

def optimize_gp(gp: GaussianProcess)-> np.ndarray:
    #nper = 1000//(len(gp.x) - 1)
    #T = np.concatenate([ np.linspace(gp.x[i-1], gp.x[i], nper) for i in range(1,len(gp.x)) ])
    T = np.linspace(gp.x.min(), gp.x.max(), 200)
    f = lambda x: gp.UCB(x, -2)[:x.shape[0]]
    g = elementwise_grad(f)#lambda x: gp.UCB(x, -2)[x.shape[0]:]#
    #gg = elementwise_grad(g)
    f0 = gp.UCB(gp.x.min(), -2)[0]
    T = np.stack((T[:-1], T[1:]), axis=1)
    rows = np.arange(T.shape[0])

    for _ in range(10):
        mi = T.mean(axis=1)
        fmi = f(mi)
        gmi = g(mi)
        if any((abs(gmi) < 1e-6) & (fmi < f0)): break
        #print(f'{g(mi)[:10] = }')
        #print(f'{gp.UCB(mi,-2)[mi.shape[0]:][:10] = }')
        T[rows,1*(gmi>0.)] = mi

    T = T.reshape(-1)

    indx = np.argmin(gp.UCB(T, -2.)[:T.shape[0]])

    return T[indx]

from scipy.optimize import minimize
    
def optimize_hyper(gp: GaussianProcess)-> np.ndarray:
    def op_fun(theta: np.ndarray)-> np.ndarray:
        mun = gp.mu.parameters.shape[0]
        gp.mu.parameters = theta[:mun]
        gp.ker.parameters = theta[mun:]
        gp.update()
        return -gp.logL()# + 1e-6*np.linalg.norm(theta[mun:])**2

    x0 = np.concatenate((gp.mu.parameters,gp.ker.parameters))
    g = jacobian(op_fun)

    from scipy.optimize import minimize
    fg = value_and_grad(op_fun)
    mun = gp.mu.parameters.shape[0]
    res = minimize(fg, x0, jac=True,options={'gtol': 1e-6, 'ftol': 0.}, bounds=np.concatenate([gp.mu.bounds, gp.ker.bounds]))
    #print(res)
    gp.mu.parameters = res.x[:mun]
    gp.ker.parameters = res.x[mun:]
    gp.update()

    return x0

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

def line_search_wolfe5(fg: Callable[[np.ndarray],tuple[float,np.ndarray]],
                       xk: np.ndarray,
                       d: np.ndarray,
                       old_fval: float=None,
                       g: np.ndarray=None,
                       c1: float=1e-4,
                       c2: float=.9,
                       amax: float=1000,
                       amin: float=1e-14,
                       old_old_fval: float=None,
                       verbose: int=0,
                       np=np)-> float:
    f_old = old_fval
    g_old = g
    phi = lambda s: fg(xk + s*d)
    stp = min(amax, 1.)
    fg_cnt = 0
    gd_old = np.dot(g, d)
    gd = gd_old
    gd_low = gd_old
    gdinit = gd_old
    gtest = c1*gd_old
    finit = f_old
    f_low = finit
    delta = 0.
    alpha = 1.

    if f_old is None or g_old is None:
        f_old, g_old = phi(0.)
        fg_cnt += 1

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
        g_low = g
        gd_low = gd
        f_low = f
        delta = delta + alpha
        alpha = 4. * alpha
    else:
        return (delta-alpha/4.) + alpha/4.*stp, fg_cnt, f, g

    stp = delta + alpha*stp
    ftest = finit + stp*gtest
    if f < ftest and abs(g.dot(d)) <= c2 * (-gdinit):
        if verbose >= 99:
            print('STRONG WOLFE SATISFIED')
        return stp, fg_cnt, f, g

    x = np.array([ delta, stp ])
    y = np.array([ f_low, f ])
    gx = np.array([ gd_low, gd ])

    a = np.linalg.lstsq([[0,0,0,1.],
                         [1,1,1,1],
                         [0,0,1,0],
                         [3,2,1,0]], [*y, *gx], rcond=-1)[0]
    
    pfg = value_and_grad(lambda x: polyval(a, x))
    res1 = minimize(pfg, delta, jac=True, options={'gtol': 1e-6, 'ftol': 1e-16}, bounds=[(delta, stp)])
    res2 = minimize(pfg, stp, jac=True, options={'gtol': 1e-6, 'ftol': 1e-16}, bounds=[(delta, stp)])

    xvals = [ *x ]
    fvals = [ *y ]
    gvals = [ g_old, g ]
    
    stp2 = (res2.x if res2.fun < res1.fun else res1.x)[0]
    stp = np.clip(stp2, (stp-delta)*.3 + delta, stp - (stp-delta)*.3)

    f, g, alpha, fg_new = zipNaN(phi, stp-delta, delta)
    fg_cnt += fg_new
    if alpha is None:
        print('No step size found')
        return None, fg_cnt, finit, g_old
    stp = delta + alpha
    
    ftest = finit + stp*gtest
    if f < ftest and abs(g.dot(d)) <= c2 * (-gdinit):
        if verbose >= 99:
            print('STRONG WOLFE SATISFIED')
        return stp, fg_cnt, f, g
    
    xvals.append(stp)
    fvals.append(f)
    gvals.append(g)
    
    mu = PolyRegressor(a)
    ker = RBF(10., 3.)#CubicSpline(1.,1.)#RBF(10., 3.)
    
    x = np.array(xvals)
    y = np.array(fvals)
    gx = np.array([ gs.dot(d) for gs in gvals ])

    try:
        gp = GaussianProcess(mu, ker, reg=np.clip(max(min(abs(y)), min(abs(gx)))*1e-1, 1e-16, 1e-10))
        gp.add(x, y, gx)
    except (np.linalg.LinAlgError, ValueError) as e:
        warnings.warn(f'Line search error: {e}')
        indx = len(fvals) - 1 - np.argmin(fvals[::-1])
    
        return xvals[indx], fg_cnt, fvals[indx], gvals[indx]
    
    default_ker = np.array([1e-10, 1e5])
    default_mu = np.linalg.lstsq(np.vander(gp.x, 4), gp.y, rcond=None)[0]#np.zeros_like(gp.mu.parameters)
    for _i in range(20):
        try:
            gp.ker.parameters = default_ker.copy()
            gp.mu.parameters = default_mu.copy()
            gp.update()
        except (np.linalg.LinAlgError, ValueError) as e:
            warnings.warn(f'Line search error: could not initialize hyperparameters')
            break
        try:
            theta = optimize_hyper(gp)
        except (np.linalg.LinAlgError, ValueError) as e:
            warnings.warn(f'Line search error: could not optimize hyperparameters')
            gp.mu.parameters = default_mu.copy()
            gp.ker.parameters = default_ker.copy()
            gp.update()
        #plot_gp(gp, lambda x: phi(x)[0], lambda x: phi(x)[1]@d)
        try:
            stp = optimize_gp(gp)
            df = gp.x - stp
            hi = np.min(df[df>0.], initial=gp.x.max()-stp)
            lo = np.max(df[df<=0.])
            stp = np.clip(stp, (hi-lo)*1e-3 + lo+stp,hi+stp-(hi-lo)*1e-3)
            f, g, alpha, fg_new = zipNaN(phi, stp - gp.x.min(), gp.x.min())
            if alpha is None:
                break
            stp_new = gp.x.min() + alpha
            fg_cnt += fg_new
            if stp_new != stp:
                indx = (gp.x < stp_new)
                stp = stp_new
                gp.x = gp.x[indx]
                gp.y = gp.y[indx]
                gp.g = gp.g[indx]
            ftest = finit + stp*gtest
            if f < ftest and abs(g.dot(d)) <= c2 * (-gdinit):
                if verbose >= 99:
                    print('STRONG WOLFE SATISFIED')
                return stp, fg_cnt, f, g
            xvals.append(stp)
            fvals.append(f)
            gvals.append(g)
            gp.ker.parameters = default_ker.copy()
            gp.mu.parameters = default_mu.copy()
            gp.add(stp, fvals[-1], np.dot(gvals[-1], d))
        except (np.linalg.LinAlgError, ValueError) as e:
            warnings.warn(f'Line search error: {e}')
            break
    
    indx = len(fvals) - 1 - np.argmin(fvals[::-1])
    x_res = xvals[indx]
    f_res = fvals[indx]
    g_res = gvals[indx]
    
    if indx == 0:
        fg_cnt += 1
        x_res = xvals[0] + amin
        f_res, g_res = phi(x_res)
    
    return x_res, fg_cnt, f_res, g_res



def backNaN(phi: Callable[[float],tuple[float,np.ndarray]],
            lo: float,
            hi: float,
           )-> tuple[float, np.ndarray, float, int]:

    fg_cnt = 0
    for _i in range(20):
        f, g = phi(hi)
        fg_cnt += 1
        if np.isneginf(f) or np.isfinite(f) and np.isfinite(g).all():
            return f, g, hi, fg_cnt
        
        hi = .5*(lo+hi)

    return f, g, None, fg_cnt


def line_search_wolfe6(fg: Callable[[np.ndarray],tuple[float,np.ndarray]],
                       xk: np.ndarray,
                       d: np.ndarray,
                       old_fval: float=None,
                       g: np.ndarray=None,
                       c1: float=1e-4,
                       c2: float=.9,
                       amax: float=1000.,
                       amin: float=0., # not used
                       old_old_fval: float=None,
                       verbose: int=0,
                       np=np)-> float:
    f_old = old_fval
    g_old = g
    phi = lambda s: fg(xk + s*d)
    stp = min(1., amax)
    fg_cnt = 0
    gd_old = np.dot(g, d)
    gd = gd_old
    gd_low = gd_old
    gdinit = gd_old
    gtest = c1*gd_old
    finit = f_old
    f_low = finit
    g_low = g_old
    lo = 0.
    hi = stp

    if f_old is None or g_old is None:
        f_old, g_old = phi(0.)
        fg_cnt += 1
        
    for _j in range(20):
        f, g, hi_new, fg_new = backNaN(phi, lo, hi)
        fg_cnt += fg_new
        if hi_new is None:
            print('No step size found')
            return None, fg_cnt, finit, g_old
        if np.isneginf(f):
            return hi_new, fg_cnt, f, g
        gd = g.dot(d)
        if f >= f_low or gd >= c2*gdinit:
            break
        g_low = g
        gd_low = gd
        f_low = f
        lo = hi
        hi *= 4.
    else:
        return hi/4., fg_cnt, f, g

    fvals = [f_low, f]
    gvals = [g_low, g]
    xvals = [lo, hi]
    
    for _i in range(20):
        if f < finit:
            break
        stp = xvals[-1]/10.
        f, g = phi(stp)
        fg_cnt += 1
        fvals.append(f)
        gvals.append(g)
        xvals.append(stp)
    else:
        return xvals[-1], fg_cnt, fvals[-1], gvals[-1]
    
    stp = hi
    ftest = finit + stp*gtest
    if f < ftest and abs(g.dot(d)) <= c2 * (-gdinit):
        if verbose >= 99:
            print('STRONG WOLFE SATISFIED')
        return stp, fg_cnt, f, g

    mu = PolyRegressor(np.zeros(4))
    ker = RBF(10., 3.)

    x = np.array(xvals)
    y = np.array(fvals)
    gx = np.array([ gs.dot(d) for gs in gvals ])

    try:
        gp = GaussianProcess(mu, ker, reg=np.clip(max(min(abs(y)), min(abs(gx)))*1e-1, 1e-16, 1e-10))
        gp.add(x, y, gx)
    except (np.linalg.LinAlgError, ValueError) as e:
        warnings.warn(f'Line search error: {e}')
        indx = len(fvals) - 1 - np.argmin(fvals[::-1])
    
        return xvals[indx], fg_cnt, fvals[indx], gvals[indx]
    
    default_ker = np.array([1e0, 1e-10])
    default_mu = np.linalg.lstsq(np.vander(gp.x, 4), gp.y, rcond=None)[0]#np.zeros_like(gp.mu.parameters)
    for _i in range(20):
        try:
            gp.ker.parameters = default_ker.copy()
            gp.mu.parameters = default_mu.copy()
            gp.update()
        except (np.linalg.LinAlgError, ValueError) as e:
            warnings.warn(f'Line search error: could not initialize hyperparameters')
            break
        try:
            theta = optimize_hyper(gp)
        except (np.linalg.LinAlgError, ValueError) as e:
            warnings.warn(f'Line search error: could not optimize hyperparameters')
            gp.mu.parameters = default_mu.copy()
            gp.ker.parameters = default_ker.copy()
            gp.update()
        #plot_gp(gp, lambda x: phi(x)[0], lambda x: phi(x)[1]@d)
        try:
            stp = optimize_gp(gp)
            df = gp.x - stp
            hi = np.min(df[df>0.], initial=gp.x.max()-stp)
            lo = np.max(df[df<=0.])
            stp = np.clip(stp, (hi-lo)*1e-3 + lo+stp,hi+stp-(hi-lo)*1e-3)
            f, g, stp_new, fg_new = backNaN(phi, gp.x.min(), stp)
            if stp_new is None:
                break
            fg_cnt += fg_new
            if stp_new != stp:
                indx = (gp.x < stp_new)
                stp = stp_new
                gp.x = gp.x[indx]
                gp.y = gp.y[indx]
                gp.g = gp.g[indx]
            ftest = finit + stp*gtest
            if f < ftest and abs(g.dot(d)) <= c2 * (-gdinit):
                if verbose >= 99:
                    print('STRONG WOLFE SATISFIED')
                return stp, fg_cnt, f, g
            xvals.append(stp)
            fvals.append(f)
            gvals.append(g)
            gp.ker.parameters = default_ker.copy()
            gp.mu.parameters = default_mu.copy()
            gp.add(stp, fvals[-1], np.dot(gvals[-1], d))
        except (np.linalg.LinAlgError, ValueError) as e:
            warnings.warn(f'Line search error: {e}')
            break
    
    indx = np.argmin(fvals)
    x_res = xvals[indx]
    f_res = fvals[indx]
    g_res = gvals[indx]
    
    return x_res, fg_cnt, f_res, g_res


if __name__ == '__main__':

    x = [[0,0,0,0,1],
         [0.3**4,.3**3,.3**2,.3,1],
         [4*.3**3, 3*.3**2, 2*.3,1,0],
         [4*.6**3, 3*.6**2, 2*.6,1,0],
         [1,1,1,1,1]]
    y = [1,0,0,0,1]
    A = np.linalg.lstsq(x, y, rcond=-1)[0]

    f = lambda x: np.exp(x)*x - np.sqrt(.01+x) + np.cos(x)#polyval(A, x)
    g = elementwise_grad(f)#polyval(A[:-1]*[4,3,2,1], x)
    T = np.linspace(0,1,10000)
    plt.plot(T, f(T))
    plt.show()


    x = np.array([0., 1.])
    y = f(x)
    gx = g(x)

    a = np.linalg.lstsq([[0,0,0,1.],
                         [1,1,1,1],
                         [0,0,1,0],
                         [3,2,1,0]], [*f(x), *g(x)], rcond=-1)[0]
    mu = PolyRegressor(a)
    ker = RBF(10., 3.)

    x = np.array([0., 1.])
    y = f(x)
    gx = g(x)

    gp = GaussianProcess(mu, ker, reg=1e-10)
    gp.add(x, y, gx)


    T = np.linspace(0,1,100)
    for _ in range(10):
        theta = optimize_hyper(gp)

        plot_gp(gp, f, g)

        nxt = optimize_gp(gp)
        t = np.linspace(0, 1, 1000)
        plt.plot(t, gp.UCB(t,-2)[:t.shape[0]], '-.')
        plt.plot([nxt], [gp.UCB(nxt,-2)[0]], 'x')
        plt.show()
        gp.add(nxt, f(nxt), g(nxt))

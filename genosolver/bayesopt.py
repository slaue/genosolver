from abc import ABC, abstractmethod
from typing import Callable, Optional
from functools import wraps
import matplotlib.pyplot as plt
from autograd import grad, elementwise_grad, jacobian, value_and_grad
import autograd.numpy as np
import autograd.scipy.stats as stats
import autograd.scipy.linalg as lin

class Optimizable(ABC):

    __slots__ = [ 'theta' ]

    theta: np.ndarray

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
        C = g0(x)
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

    def __init__(self, sig: float=1.5, theta: float=.1):
        self.theta = np.array([ sig, theta ])

    def __call__(self, d: np.ndarray)-> np.ndarray:
        xdr = self.theta[1]**2*np.sqrt(5)*abs(d)
        return self.theta[0]**2*(1 + xdr + xdr**2/3)*np.exp(-xdr)

    def grad(self, x: np.ndarray)-> np.ndarray:
        y = super().grad(x)
        c = self.theta[0]**2*self.theta[1]**4*5/3
        mask = np.block([ [np.zeros_like(x), np.zeros_like(x)], [np.zeros_like(x), x==0] ])
        y = y + c * mask
        return y

class RBF(Kernel):
    '''
    Gaussian Kernel
    '''
    
    def __init__(self, alpha: float=10., theta: float=.1):
        self.theta = np.array([ alpha, theta ])

    def __call__(self, d: np.ndarray)-> np.ndarray:
        return self.theta[0]**2 * np.exp(-self.theta[1]**2*(d)**2)

class RQK(Kernel):
    '''
    Rational Quadratic Kernel
    '''

    def __init__(self, alpha: float=10., theta: float=1., p: float=1.):
        self.theta = np.array([ alpha, theta, p ])

    def __call__(self, d: np.ndarray)-> np.ndarray:
        return self.theta[0]**2*(1+(self.theta[1]*d*self.theta[2])**2)**(-self.theta[2]**2)

def polyval(p, x):
    mx = 0.
    for a in p:
        mx = a + mx * x
    return mx

class PolyRegressor(Expectation):
    '''
    Polynomial regression
    '''

    def __init__(self, p: np.ndarray):
        self.theta = p

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
        cov = self.ker(self.x[:,None]-x) if self.g is None else self.ker.grad(self.x[:,None]-x)
        diff = (self.y - self.mu(self.x)) if self.g is None else (np.concatenate((self.y, self.g)) - self.mu.grad(self.x))
        v = lin.solve_triangular(self._L, diff, lower=True)
        wt = lin.solve_triangular(self._L, cov, lower=True)
        mux = self.mu(x) if self.g is None else self.mu.grad(self.x)
        return mux + wt.T @ v

    def covary(self, x: np.ndarray)-> np.ndarray:
        x = np.atleast_1d(x)
        cov = self.ker(self.x[:,None]-x) if self.g is None else self.ker.grad(self.x[:,None]-x)
        V = lin.solve_triangular(self._L, cov, lower=True)
        kro = self.ker(x[:,None]-x) if self.g is None else self.ker.grad(x[:,None]-x)
        return kro - V.T @ V

    def predict(self, x: np.ndarray)-> tuple[np.ndarray, np.ndarray]:
        x = np.atleast_1d(x)
        cov = self.ker(self.x[:,None]-x) if self.g is None else self.ker.grad(self.x[:,None]-x)
        diff = (self.y - self.mu(self.x)) if self.g is None else (np.concatenate((self.y, self.g)) - self.mu.grad(self.x))
        V = lin.solve_triangular(self._L, cov, lower=True)
        w = lin.solve_triangular(self._L, diff, lower=True)
        mux = self.mu(x) if self.g is None else self.mu.grad(x)
        kro = self.ker(x[:,None]-x) if self.g is None else self.ker.grad(x[:,None]-x)
        return mux + V.T @ w, kro - V.T @ V

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
        self.x = np.block([ self.x, x ])
        self.y = np.block([ self.y, y ])
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
        diff = (self.y - self.mu(self.x)) if self.g is None else (np.concatenate((self.y, self.g)) - mu_grad(self.mu)(self.x))
        z = lin.solve_triangular(self._L, diff, lower=True)
        return -np.sum(np.log(np.diag(self._L))) - .5 * np.dot(z.T, z) - .5 * self.x.shape[0]*np.log(2.*np.pi)

def plot_gp(gp: GaussianProcess, f=None, g=None):
    n = 1000
    t = np.linspace(0, 1, 1000)
    ex, cov = gp.predict(t)
    if gp.g is None:
        if f is not None:
            plt.plot(t, f(t), '-.')
        plt.plot(t, ex)
        plt.plot(t, ex + 2 * np.diagonal(cov), '--r')
        plt.plot(t, ex - 2 * np.diagonal(cov), '--r')
        plt.plot(gp.x, gp.y, 'x')
        plt.show()
    else:
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
        if f is not None:
            axs[0].plot(t, f(t), '-.')
        axs[0].plot(t, ex[:n])
        axs[0].plot(t, ex[:n] + 2 * np.diagonal(cov[:n]), '--r')
        axs[0].plot(t, ex[:n] - 2 * np.diagonal(cov[:n]), '--r')
        axs[0].plot(gp.x, gp.y, 'x')
        if g is not None:
            axs[1].plot(t, g(t), '-.')
        axs[1].plot(t, ex[n:])
        axs[1].plot(t, ex[n:] + 2 * np.diagonal(cov[n:]), '--r')
        axs[1].plot(t, ex[n:] - 2 * np.diagonal(cov[n:]), '--r')
        axs[1].plot(gp.x, gp.g, 'x')
        plt.show()
        
def optimize_gp(gp: GaussianProcess)-> np.ndarray:
    T = np.linspace(0., 1., 100)
    f = lambda x: gp.UCB(x, -2)[:x.shape[0]]
    g = elementwise_grad(f)
    #gg = elementwise_grad(g)

    for _ in range(5):
        s = np.ones(T.shape[0])
        while np.any(idx := (f(T) < f(T - s*g(T)))):
            s[idx] *= .5
        T = T - s*g(T)
        T = np.clip(T, 0., 1.)

    indx = np.argmin(gp.UCB(T, -2.)[:T.shape[0]])

    return T[indx]

def optimize_hyper(gp: GaussianProcess)-> np.ndarray:
    def op_fun(theta: np.ndarray)-> np.ndarray:
        mun = gp.mu.parameters.shape[0]
        gp.mu.parameters = theta[:mun]
        gp.ker.parameters = theta[mun:]
        gp.update()
        return -gp.logL()# + 1e-4*np.linalg.norm(theta[mun:])**4

    x0 = np.concatenate((gp.mu.parameters,gp.ker.parameters))
    g = jacobian(op_fun)
    
    from scipy.optimize import minimize
    fg = value_and_grad(op_fun)
    res = minimize(fg, x0, jac=True, method='CG', options={'gtol': 1e-6})
    mun = gp.mu.parameters.shape[0]
    gp.mu.parameters = res.x[:mun]
    gp.ker.parameters = res.x[mun:]
    gp.update()
        
    return x0

def line_search_wolfe5(fg: Callable[[np.ndarray],tuple[float,np.ndarray]],
                       x: np.ndarray,
                       d: np.ndarray,
                       old_fval: float=None,
                       g: np.ndarray=None,
                       c1: float=1e-4,
                       c2: float=.9,
                       amax: float=1000,
                       amin: float=1e-14,
                       verbose: int=0)-> float:
    f_old = old_fval
    g_old = g
    phi = lambda s: fg(x + s*d)
    stp = min(amax, 1.)
    fg_cnt = 0
    gd_old = np.dot(g, d)
    gtest = c1*gd_old
    finit = f_old
    f_low = finit
    delta = 0.
    alpha = 1.
    
    if f_old is None or g_old is None:
        f_old, g_old = phi(0.)
        fg_cnt += 1

    for _j in range(20):
        for _i in range(20):
            f, g = phi(delta + alpha*stp)
            fg_cnt += 1
            if np.isneginf(f):
                break
            if np.isfinite(f) and np.isfinite(g).all():
                break

            if verbose >= 99:
                print('f or g has inf or nan')

            stp = .5 * stp
            alpha = .5 * alpha
        else:
            print('No step size found')
            return None, fg_cnt, finit, g_old
        if f >= f_low or g.dot(d) >= c2*gdinit:
            break
        g_low = g
        gd_low = g.dot(d)
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
    mu = PolyRegressor(a)
    ker = RBF(10., 3.)

    gp = GaussianProcess(mu, ker, reg=1e-8)
    gp.add(x, y, gx)

    xvals = [ *x ]
    fvals = [ *y ]
    gvals = [ g_old, g ]

    for _ in range(20):

        ftest = finit + stp*gtest
        if f < ftest and abs(g.dot(d)) <= c2 * (-gdinit):
            if verbose >= 99:
                print('STRONG WOLFE SATISFIED')
            return stp, fg_cnt, f, g
        
        gp.ker.parameters = np.array([5.,3.])
        theta = optimize_hyper(gp)
        stp = optimize_gp(gp)
        f, g = phi(stp)
        fg_cnt += 1
        xvals.append(stp)
        fvals.append(f)
        gvals.append(g)
        gp.add(stp, fvals[-1], np.dot(gvals[-1], d))

    indx = np.argmin(fvals)
        
    return xvals[indx], fg_cnt, fvals[indx], gvals[indx]
    
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

    gp = GaussianProcess(mu, ker, reg=1e-6)
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

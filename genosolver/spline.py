#from __future__ import annotations
#from functools import total_ordering
from dataclasses import dataclass, field
from typing import Callable, Optional
from queue import PriorityQueue
from bisect import bisect_left, insort_left
import numpy as np # for np.exp and np.linspace

@dataclass(slots=True, order=True)
class Cubic:

    """
    Fitting a cubic polynomial.
    Transfering [x0,x1] to [0,1], solving for coefficients, and then rescaleing back to [x0,x1].
    When sorting, assuming all Cubics are of the same function.
    """

    x0: float
    x1: float
    a: float
    b: float
    c: float
    d: float
    _min_fx: float
    _min_x: float
    alpha: float
    gamma: float

    def __init__(self, x0: float, x1: float,
                 fx0: float, gx0: float,
                 fx1: float, gx1: float,
                 alpha: Optional[float]=None,
                 gamma: float=1.):

        gx0 = gx0 * (x1 - x0)
        gx1 = gx1 * (x1 - x0)
        self.x0 = x0
        self.x1 = x1
        self.c = gx0
        self.d = fx0
        self.a = gx0 + gx1 - 2*(fx1 - fx0)
        self.b = 3*(fx1 - fx0) - gx1 - 2*gx0
        self.alpha = 0. if alpha is None else alpha
        self.gamma = gamma
        
        self._min_fx = fx1
        self._min_x = x1

        if self._min_fx > fx0:
            self._min_fx = fx0
            self._min_x = x0

        if self.a != 0:
            sq = self.b**2 - 3*self.a*self.c
            if sq >= 0:
                sq = sq**.5
                st1 = (-self.b + sq)/3/self.a
                st2 = (-self.b - sq)/3/self.a

                if 0 <= st1 <= 1:
                    stf1 = self.nonscale_f(st1)
                    if self._min_fx > stf1:
                        self._min_fx = stf1
                        self._min_x = st1 * (self.x1 - self.x0) + self.x0
                if 0 <= st2 <= 1:
                    stf2 = self.nonscale_f(st2)
                    if self._min_fx > stf2:
                        self._min_fx = stf2
                        self._min_x = st2 * (self.x1 - self.x0) + self.x0
        elif self.b != 0:
            st1 = -self.c/self.b/2
            if 0 <= st1 <= 1:
                stf1 = self.nonscale_f(st1)
                if self._min_fx > stf1:
                    self._min_fx = stf1
                    self._min_x = st1 * (self.x1 - self.x0) + self.x0

        if not alpha is None:
            t = np.linspace(0., 1., 100)
            fmin, tmin = min(( (self.nonscale_f(i), -i) for i in t ))
            self._min_fx = fmin
            self._min_x = tmin * (self.x0 - self.x1) + self.x0
            
    def nonscale_conf_f(self, x: float)-> float:
        return self.alpha * (np.exp(-.25 * self.gamma) - np.exp(-((x - .5)**2)*self.gamma))

    def conf_f(self, x: float)-> float:
        xt = (x - slef.x0) / (self.x1 - self.x0)
        return self.nonscale_conf_f(xt)

    def nonscale_f(self, x: float)-> float:
        return ((self.a*x + self.b)*x + self.c)*x + self.d + self.nonscale_conf_f(x)

    def f(self, x: float)-> float:
        xt = (x - self.x0) / (self.x1 - self.x0)
        return self.nonscale_f(xt)

    def forward(self, x: float)-> float:
        return self.f(x)

    def __call__(self, x: float)-> float:
        return self.f(x)

    def g(self, x: float)-> float:
        xt = (x - self.x0) / (self.x1 - self.x0)
        return ((3*self.a*xt + 2*self.b)*xt + self.c) / (self.x1 - self.x0)

    @property
    def min(self)-> tuple[float, float]:
        return (self._min_x, self._min_fx)

@dataclass(slots=True)
class Spline:
    """
    Cubic spline interpolation of function.
    """

    fg: Optional[Callable[[float], tuple[float, float]]] = None
    cubics: list[Cubic] = field(init=False, default_factory=list)
    _pq: PriorityQueue[tuple[float, float, Cubic]] = field(default_factory=PriorityQueue)

    @property
    def min(self)-> Cubic:
        return self._pq.queue[0][2]

    def put(self, cub: Cubic):
        cx, fcx = cub.min
        insort_left(self.cubics, cub)
        self._pq.put((fcx, -cx, cub))

    def split_min(self):
        cub = self._pq.get()[2]
        xm = cub._min_x
        x0 = cub.x0
        x1 = cub.x1
        fxm, gxm = self.fg(xm)

        indx = bisect_left(self.cubics, cub)
        del self.cubics[indx]

        cubl = Cubic(x0, xm, cub.f(x0), cub.g(x0), fxm, gxm, alpha=1.*(1+abs(cub.f(x0)-fxm))*(xm - x0), gamma=.5)
        cubr = Cubic(xm, x1, fxm, gxm, cub.f(x1), cub.g(x1), alpha=1.*(1+abs(fxm-cub.f(x1)))*(x1 - xm), gamma=.5)

        self.put(cubl)
        self.put(cubr)

    def f(self, x: float)-> float:
        indx = bisect_left(self.cubics, x, key=lambda cub: cub.x1)
        if indx >= len(self): indx = -1
        return self.cubics[indx].f(x)

    def forward(self, x: float)-> float:
        return self.f(x)

    def __call__(self, x: float)-> float:
        return self.f(x)

    def g(self, x: float)-> float:
        indx = bisect_left(self.cubics, x, key=lambda cub: cub.x1)
        if indx >= len(self): indx = -1
        return self.cubics[indx].g(x)

    def __getitem__(self, indx: int)-> Cubic:
        return self.cubics[indx]

    def __len__(self)-> int:
        return self.cubics.__len__()

import numpy

def line_search_wolfe4(fg, xk, d, g=None,
                       old_fval=None, old_old_fval=None,
                       args=(), c1=1e-4, c2=0.9, amax=50., amin=1e-14,
                       xtol=1e-14, verbose=0, np=numpy):
    
    stp = np.clip(1., amin, amax)

    delta = 0.
    alpha = 1.
    def phi(s):
        fx, gx = fg(xk + (delta + alpha*s)*d)
        return fx, gx

    fg_cnt = 0
    if old_fval is None or g is None:
        old_fval, g = phi(0)
        fg_cnt += 1
    
    eps = 0.0
    finit = old_fval
    gdinit = g.dot(d)
    gtest = c1*gdinit
    g_old = g
    
    Q = PriorityQueue()

    for _j in range(20):
        for _i in range(20):
            f, g = phi(stp)
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
        if g.dot(d) >= c2*gdinit:
            break
        delta = delta + alpha
        alpha = 4. * alpha
    else:
        return delta + alpha*stp, fg_cnt, f, g
    
    gd = g.dot(d)
    cub = Cubic(0, stp, finit, gdinit, f, gd)#, alpha=(1.+abs(f-finit))*stp, gamma=.5)
    xm, fxm = cub.min
    Q.put((fxm, -xm, cub))

    best_f = f
    best_g = g
    best_stp = stp
    
    for _i in range(20):
        
        ftest = finit + stp*gtest
        if f < ftest and abs(gd) <= c2 * (-gdinit):
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
        if not (x0 < stp < x1):
            stp = np.clip(stp, x0 + (x1 - x0) * 1e-6, x1 - (x1 - x0) * 1e-6)
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
        
        cubl = Cubic(x0, stp, cub.f(x0), cub.g(x0), f, g.dot(d))#, alpha=(1.+abs(f-cub.f(x0)))*(stp-x0), gamma=.5)
        xl, fxl = cubl.min
        Q.put((fxl, -xl, cubl))

        if stp < x1:
            cubr = Cubic(stp, x1, f, g.dot(d), cub.f(x1), cub.g(x1))#, alpha=(1.+abs(f-cub.f(x1)))*(x1-stp), gamma=.5)
            xr, fxr = cubr.min
            Q.put((fxr, -xr, cubr))
    else:
        if verbose >= 99:
            print('MAX ITER line search')

    #if best_stp > amin:
    #    best_stp = amin
    #    fg_cnt += 1
    #    best_f, best_g = phi(amin)
    
    return delta + alpha*best_stp, fg_cnt, best_f, best_g

def line_search_wolfe4_debug(fg, xk, d, g=None,
                             old_fval=None, old_old_fval=None,
                             args=(), c1=1e-4, c2=0.9, amax=50., amin=1e-14,
                             xtol=1e-14, verbose=100, np=numpy, plot_path=None):
    
    stp = np.clip(1., amin, amax)

    steps_array = []
    save_steps = True
    def phi(s):
        if save_steps:
            steps_array.append(s)
        fx, gx = fg(xk + s*d)
        return fx, gx

    fg_cnt = 0
    if old_fval is None or g is None:
        old_fval, g = phi(0)
        fg_cnt += 1
    
    eps = 0.0
    finit = old_fval
    gdinit = g.dot(d)
    gtest = c1*gdinit
    
    def phid(s):
        f, g = phi(s)
        return f, g.dot(d)
    S = Spline(phid)

    f, g = phi(stp)
    gd = g.dot(d)
    fg_cnt += 1
    cub = Cubic(0, stp, finit, gdinit, f, g.dot(d), alpha=(1.+abs(f-finit))*stp, gamma=.5)
    S.put(cub)

    best_f = f
    best_g = g
    best_stp = stp
    
    for _ in range(20):
        ftest = finit + stp*gtest
        if (f < ftest + eps*(abs(ftest) + 1) and abs(gd) <= c2 * (-gdinit)):
            best_f = f
            best_g = g
            best_stp = stp
            break

        cub = S.min
        x0 = cub.x0
        x1 = cub.x1
        stp = cub.min[0]
        if not (x0 < stp < x1) or (x1 - x0) < xtol * x1:
            if verbose > 9:
                print('XTOL satisfied')
            break

        S.split_min()
        fg_cnt += 1
        #f = S.f(stp)
        #g = S.g(stp)
        f, g = phi(stp)

        if (f, g.dot(d), -stp) < (best_f, best_g.dot(d), -best_stp):
            best_f = f
            best_g = g
            best_stp = stp
        
    if stp:
        steps_array.append(stp)
    if stp != 1.:
        import matplotlib.pyplot as plt
        import os, datetime
        save_steps = False
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14,6))
        step_space = np.linspace(0., max(steps_array), 50)
        axs[0].plot(step_space, [ phi(s)[0] for s in step_space ])
        axs[0].plot(steps_array, [ phi(s)[0] for s in steps_array], '.')
        axs[0].plot(step_space, [ S.f(s) for s in step_space ], '--g')
        axs[1].semilogy(list(range(len(steps_array))), steps_array)
        if not plot_path is None: 
          os.makedirs(plot_path, exist_ok=True)
          plot_name = datetime.datetime.now().strftime('%Y-%m-%d-%H_%M_%S')
          plt.save(os.path.join(plot_path, f'{plot_name}.pdf'))
        else:
            plt.show()
        plt.close()

    #if best_stp > amin:
    #    best_stp = amin
    #    fg_cnt += 1
    #    best_f, best_g = phi(amin)

    return best_stp, fg_cnt, best_f, best_g


    
if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from pprint import pprint
    '''
    t = np.linspace(0, 2.5, 100)
    plt.plot([.5, 2], [1, -1], '.r')
    cub = Cubic(.5, 2, 1, -2, -1, -1)
    plt.plot(t, cub(t))
    for m in [0., 1., 2., 5., 10.]:
        cub = Cubic(.5, 2, 1, -2, -1, -1, alpha=1., gamma=m+1)
        plt.plot(t, cub(t), '--')
        plt.plot(*cub.min, 'x')
    plt.show()

    t = np.linspace(0, 2.5, 100)
    plt.plot([.5, 2], [0, 0], '.r')
    cub = Cubic(.5, 2, 0, 0, 0, 0)
    plt.plot(t, cub(t))
    for m in [0., 1., 2., 5., 10.]:
        cub = Cubic(.5, 2, 0, 0, 0, 0, alpha=1., gamma=m+1)
        plt.plot(t, cub(t), '--')
        plt.plot(*cub.min, 'x')
    plt.show()

    t = np.linspace(0, 2.5, 100)
    plt.plot([.5, 2], [0, 1], '.r')
    cub = Cubic(.5, 2, 0, -2, 1, -3)
    plt.plot(t, cub(t))
    for m in [0., 1., 2., 5., 10.]:
        cub = Cubic(.5, 2, 0, -2, 1, -3, alpha=1., gamma=m+1)
        plt.plot(t, cub(t), '--')
        plt.plot(*cub.min, 'x')
    plt.show()

    t = np.linspace(-.5, 1.5, 100)
    plt.plot([0, 1], [1, 1], '.r')    
    cub = Cubic(0, 1, 1, -2, 1, 2)
    plt.plot(t, cub(t))
    for m in [0., 1., 2., 5., 10.]:    
        cub = Cubic(0, 1, 1, -2, 1, 2, alpha=1., gamma=m+1)
        plt.plot(t, cub(t), '--')
        plt.plot(*cub.min, 'x')
    plt.show()

    t = np.linspace(-.5, 1.5, 100)
    plt.plot([0, 1], [0, 0], '.r')    
    cub = Cubic(0, 1, 0, -1, 0, 2)
    plt.plot(t, cub(t))
    for m in [0., 1., 2., 5., 10.]:    
        cub = Cubic(0, 1, 0, -1, 0, 2, alpha=m)
        plt.plot(t, cub(t), '--')
        plt.plot(*cub.min, 'x')
    plt.show()

    t = np.linspace(.5, 2.5, 100)
    plt.plot([1, 2], [0, 0], '.r')    
    cub = Cubic(1, 2, 0, -1, 0, 2)
    plt.plot(t, cub(t))
    for m in [0., 1., 2., 5., 10.]:    
        cub = Cubic(1, 2, 0, -1, 0, 2, alpha=m)
        plt.plot(t, cub(t), '--')
        plt.plot(*cub.min, 'x')
    plt.show()

    t = np.linspace(1, 5, 100)
    plt.plot([2, 4], [0, 0], '.r')
    cub = Cubic(2, 4, 0, -1, 0, 2)
    plt.plot(t, cub(t))
    for m in [0., 1., 2., 5., 10.]:    
        cub = Cubic(2, 4, 0, -1, 0, 2, alpha=m)
        plt.plot(t, cub(t), '--')
        plt.plot(*cub.min, 'x')
    plt.show()

    
    f = lambda x: x**3 - 2*x**2 + 3
    g = lambda x: 3*x**2 - 4*x
    cub = Cubic(1, 3, f(1), g(1), f(3), g(3))
    res, resf = cub.min
    #print(f'{res, g(res), cub.g(res) = }')
    
    t = np.linspace(-3, 3, 100)
    plt.plot(t, f(t), '-k')
    plt.plot(t, cub(t), '--r')
    plt.plot([1, 3], [f(1), f(3)], '.b')
    plt.plot(*cub.min, 'xb')
    plt.show()

    plt.plot(t, g(t), '-k')
    plt.plot(t, cub.g(t), '--r')
    plt.show()

    spl = Spline(lambda x: (np.cos(x), -np.sin(x)))
    
    t = np.linspace(-.5, 30.5, 100)
    spl.put(Cubic(0, t[-1]-.5, 1, 0, np.cos(t[-1]-.5), -np.sin(t[-1]-.5)))

    for _ in range(5):
        plt.plot(t, np.cos(t), '-k')
        plt.plot(t, [ spl(x) for x in t ], '--r')
        plt.show()
        print(spl.min)
        spl.split_min()


    f = np.abs
    g = np.sign

    spl = Spline(lambda x: (f(x), g(x)))
    t = np.linspace(-1.5, 3.5, 100)
    spl.put(Cubic(t[0]+.5, t[-1]-.5, f(t[0]+.5), g(t[0]+.5), f(t[-1]-.5), g(t[-1]-.5)))

    for _ in range(5):
        plt.plot(t, f(t), '-k')
        plt.plot(t, [ spl(x) for x in t ], '--r')
        plt.plot([ x.x0 for x in spl.cubics ], [ x(x.x0) for x in spl.cubics ], '.b')
        plt.plot([ spl.cubics[-1].x1 ], [ spl.cubics[-1](spl.cubics[-1].x1) ], '.b')
        plt.plot([ x._min_x for x in spl.cubics ], [ x._min_fx for x in spl.cubics ], 'xb')
        plt.show()
        print(spl.min)
        spl.split_min()

    def fg(x):
        if x < 1: return (1-x, -1)
        return (np.exp(1-x)-1, -np.exp(1-x))

    spl = Spline(fg)
    
    t = np.linspace(-.5, 4.5, 100)
    spl.put(Cubic(t[0]+.5, t[-1]-.5, *fg(t[0]+.5), *fg(t[-1]-.5)))

    for _ in range(1):
        plt.plot(t, [ fg(x)[0] for x in t ], '-k')
        plt.plot(t, [ spl(x) for x in t ], '--r')
        plt.plot([ x.x0 for x in spl.cubics ], [ x(x.x0) for x in spl.cubics ], '.b')
        plt.plot([ spl.cubics[-1].x1 ], [ spl.cubics[-1](spl.cubics[-1].x1) ], '.b')
        plt.plot([ x._min_x for x in spl.cubics ], [ x._min_fx for x in spl.cubics ], 'xb')
        plt.show()
        print(spl.min)
        spl.split_min()
    '''
    def fg(x):
        if x < 1: return ((1-x)*100, -1*100)
        def sigmoid(x):
          return 1 / (1 + np.exp(-x))
        return ((1-x)*sigmoid(1-x)*100, -sigmoid(1-x)*100 + -(1-x)*sigmoid(1-x)*(1-sigmoid(1-x))*100)

    
    spl = Spline(fg)
    
    t = np.linspace(-.5, 4.5, 100)
    spl.put(Cubic(t[0]+.5, t[-1]-.5, *fg(t[0]+.5), *fg(t[-1]-.5), alpha=1., gamma=1.))
    
    from scipy.optimize import line_search

    for _ in range(10):
        plt.plot(t, [ fg(x)[0] for x in t ], '-k')
        plt.plot(t, [ spl(x) for x in t ], '--r')
        plt.plot([ x.x0 for x in spl.cubics ], [ x(x.x0) for x in spl.cubics ], '.b')
        plt.plot([ spl.cubics[-1].x1 ], [ spl.cubics[-1](spl.cubics[-1].x1) ], '.b')
        plt.plot([ x._min_x for x in spl.cubics ], [ x._min_fx for x in spl.cubics ], 'xb')
        stps = []
        def gg(x):
            stps.append(x)
            return fg(x)[1]
        stp, *_ = line_search(lambda x: fg(x)[0], gg, 0, 4, c2=0.)
        plt.plot(stps, [ fg(s)[0] for s in stps ], 'dg')
        plt.show()
        print(spl.min)
        spl.split_min()


    def fg(x):
        
        t_0 = np.sin(x)
        t_1 = np.abs(t_0)
        t_2 = (x - 1)
        t_3 = ((1 + x) + np.abs(t_2))
        t_4 = np.cos(x)
        t_5 = (np.exp(t_4))
        t_6 = (x ** 3)
        t_7 = (t_5 + (t_6 * t_1))
        t_8 = (t_3 ** 2)
        functionValue = (t_7 / t_3)
        gradient = ((((((3 * (x ** 2)) * t_1) / t_3) - ((t_5 * (t_0)) / t_3)) + (((t_6 * t_4) * np.sign(t_0)) / t_3)) - ((t_7 / t_8) + ((np.sign(t_2) * t_7) / t_8)))

        return functionValue, gradient
    
    spl = Spline(fg)
    
    t = np.linspace(-.5, 20.5, 100)
    spl.put(Cubic(0, t[-1]-.5, *fg(0), *fg(t[-1]-.5)))

    for _ in range(10):
        plt.plot(t, fg(t)[0], '-k')
        plt.plot(t, [ spl(x) for x in t ], '--r')
        plt.plot([ x.x0 for x in spl.cubics ], [ x(x.x0) for x in spl.cubics ], '.b')
        plt.plot([ spl.cubics[-1].x1 ], [ spl.cubics[-1](spl.cubics[-1].x1) ], '.b')
        plt.plot([ x._min_x for x in spl.cubics ], [ x._min_fx for x in spl.cubics ], 'xb')
        plt.show()
        print(spl.min)
        spl.split_min()

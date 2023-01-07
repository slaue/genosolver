# -*- coding: utf-8 -*-

"""
    GENO is a solver for non-linear optimization problems.
    It can solve constrained and unconstrained problems.
    It is written fully in Python with no dependencies and
    can run on the CPU and on the GPU.
    It can solve problems of the form:

    min_x f(x)
    s.t.  cl <= g(x) <= cu
          lb <= x <= ub

    See https://www.geno-project.org for an easy-to-use interface.


    Copyright (C) 2021-2022 Soeren Laue, Mark Blacher

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.

    Contact the developer:

    E-mail: soeren.laue@uni-jena.de
    Web:    https://www.geno-project.org
"""

import warnings

class OptimizeResult(dict):
    """
    Dictionary that returns the result of an optimization process. It is
    basically just a copy of the SciPy interface such that they can be used
    interchangeably.
    """
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return '\n'.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in sorted(self.items())])
        return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys())


class LBFGSB:
    """
    A quasi-Newton solver for solving bound-constrained optimization problems
    of the form

        min_x f(x)
        s.t.  lb <= x <= ub

    It uses the limited memory L-BFGS formula for approximating the
    Hessian of f. It avoids the inherently sequential Cauchy-point computation
    of the original L-BFGS-B solver and hence, it can be run effciently
    on the CPU and on the GPU. The algorithm is described and analyzed in [1].

    References:
        [1] Soeren Laue, Mark Blacher, and Joachim Giesen.
            Optimization for Classical Machine Learning Problems on the GPU.
            In AAAI 2022.
    """
    def __init__(self, fg, x0, np, lb=None, ub=None, options=None):
        if options is None:
            options = {}
        self.fg = fg
        self.x = x0
        self.np = np
        self.n = len(self.x)
        self.constrained = not (lb is None and ub is None)
        self.lb = lb if not lb is None else np.full(self.n, -np.inf)
        self.ub = ub if not ub is None else np.full(self.n, np.inf)
        self.set_options(options)
        self.init_matrices()
        self.working = np.full(self.n, 1.0)

    def all_options(self):
        return {'verbose', 'max_iter', 'step_max', 'max_ls',
                'eps_pg', 'm', 'grad_test', 'ls'}

    def set_options(self, options):
        unsupported = [opt for opt in options.keys() if opt not in self.all_options()]
        for opt in unsupported:
            warnings.warn(f"Option '{opt}' is not supported.", RuntimeWarning)

        self.param = options
        self.param.setdefault('verbose', 0)
        self.param.setdefault('max_iter', 1000)
        self.param.setdefault('step_max', 1E10)
        self.param.setdefault('max_ls', 30)
        self.param.setdefault('eps_pg', 1E-5)
        self.param.setdefault('m', 10)
        self.param.setdefault('grad_test', False)
        self.param.setdefault('ls', 0)
        self.max_m = self.param['m']

    def init_matrices(self):
        np = self.np
        self.storage_idx = 0
        self.S = np.zeros((0, self.n))
        self.Y = np.zeros((0, self.n))
        self.storage_S = np.empty((2 * self.max_m, self.n))
        self.storage_Y = np.empty((2 * self.max_m, self.n))

    def add_corrections(self, s, y):
        if self.storage_idx >= 2 * self.max_m:
            # move everything upfront
            self.storage_S[:self.max_m, :] = self.storage_S[self.max_m:, :]
            self.storage_Y[:self.max_m, :] = self.storage_Y[self.max_m:, :]
            self.storage_idx = self.max_m
        self.storage_S[self.storage_idx, :] = s
        self.storage_Y[self.storage_idx, :] = y

        self.storage_idx += 1
        self.S = self.storage_S[max(0, self.storage_idx - self.max_m):self.storage_idx, :]
        self.Y = self.storage_Y[max(0, self.storage_idx - self.max_m):self.storage_idx, :]

    def force_bounds(self, x):
        np = self.np
        x = np.maximum(np.minimum(x, self.ub), self.lb)
        return x

    def proj_grad_norm(self, x, g):
        np = self.np
        eps = 1E-10
        if self.constrained:
            self.working = np.full(self.n, 1.0)
            self.working[(x < self.lb + eps * 2) & (g >= 0)] = 0
            self.working[(x > self.ub - eps * 2) & (g <= 0)] = 0
        pg = np.linalg.norm(np.minimum(np.maximum(x - g, self.lb), self.ub) - x, np.inf)
        return pg

    def max_step_size(self, x, d):
        np = self.np
        if self.constrained:
            step_ub = np.full(self.n, np.inf)
            step_lb = np.full(self.n, np.inf)
            idx_ub = np.where(d > 0)
            idx_lb = np.where(d < 0)
            step_ub[idx_ub] = np.divide(self.ub[idx_ub] - x[idx_ub], d[idx_ub])
            step_lb[idx_lb] = np.divide(self.lb[idx_lb] - x[idx_lb], d[idx_lb])
            step_max = min(np.min(step_ub), np.min(step_lb))
        else:
            step_max = np.inf
        return step_max

    def line_search(self, x_old, d, step_max, f_old, g_old, quadratic):
        np = self.np
        alpha = 0.1
        beta = 0.5
        step = min(float(step_max), 1.0)
        k = 0
        fun_eval = 0
        dg = np.dot(g_old, d)
        if not dg < 0:
            print(dg)
            print(d)
            assert False

        seen_quadratic = False
        while True:
            x = x_old + step * d
            f, g = self.fg(x)
            fun_eval += 1
            # make sure that function is really quadratic when said so
            if seen_quadratic:
                if np.abs(np.dot(d, g)) > 1E-5:
                    if self.param['verbose'] >= 10:
                        print('Function is not quadratic. Use parameter ls=0 instead.')
                    quadratic = False
                    self.param['ls'] = 0

            if f <= f_old + alpha * step * dg:
                break
            if k > self.param['max_ls']:
                break
            # quadratic interpolation
            if quadratic:
                a = np.dot(g_old, d)
                b = np.dot(g, d)
                step *= a / (a - b)
                seen_quadratic = True
            else:
                step *= beta
            k += 1

        return f, g, x, step, fun_eval

    def two_loop(self, g):
        np = self.np
        k, _ = self.S.shape
        rho = np.empty(k)
        alpha = np.empty(k)
        if self.constrained:
            Yw = self.Y * self.working
            q = g * self.working
        else:
            Yw = self.Y
            q = g.copy()

        if k == 0:
            return q

        for i in range(k - 1, -1, -1):
            rho[i] = np.dot(self.S[i], Yw[i])
            if rho[i] > 1E-10:
                alpha[i] = np.dot(self.S[i], q) / rho[i]
                q -= alpha[i] * Yw[i]

        if rho[k - 1] > 1E-10:
            gamma = rho[k - 1] / np.linalg.norm(Yw[k - 1]) ** 2
            q *= gamma

        for i in range(k):
            if rho[i] > 1E-10:
                beta = np.dot(Yw[i], q) / rho[i]
                q += (alpha[i] - beta) * self.S[i]

        if self.constrained:
            q = q * self.working
        return q

    def project_direction(self, x, g, d):
        np = self.np
        eps = 1E-10
        x_new = x + d
        idx_lb = x_new < self.lb + 2 * eps
        idx_ub = x_new > self.ub - 2 * eps
        x_new[idx_lb] = self.lb[idx_lb]
        x_new[idx_ub] = self.ub[idx_ub]
        d_new = x_new - x
        if np.dot(g, d_new) < -eps:
            return d_new

        d[(d < 0) & (x <= self.lb + 2 * eps)] = 0
        d[(d > 0) & (x >= self.ub - 2 * eps)] = 0

        return d

    def num_cors(self):
        k, _ = self.S.shape
        return k

    def grad_test(self, x):
        np = self.np
        t = 1E-6
        delta = np.random.randn(self.n)
        f1, _ = self.fg(x + t * delta)
        f2, _ = self.fg(x - t * delta)
        _, g = self.fg(x)
        d = (f1 - f2) / (2 * t) - np.dot(g, delta)
        print(f'gradient test: approximation error {d:.5g}')
        return d

    def minimize(self):
        np = self.np
        eps = 1E-10
        # check for feasibility
        if np.any(self.lb > self.ub):
            return OptimizeResult(x=self.x, fun=None, jac=None,
                                  nit=0, nfev=0,
                                  status=1, success=False,
                                  message="Infeasible")
        x = self.force_bounds(self.x)
        if self.param['grad_test']:
            self.grad_test(x)

        f, g = self.fg(x)
        fun_eval = 1
        x_old = x
        g_old = g
        pg = self.proj_grad_norm(x, g)

        # check for early stopping
        if pg <= self.param['eps_pg']:
            return OptimizeResult(x=x, fun=f, jac=g,
                                  nit=0, nfev=fun_eval, status=0, success=True,
                                  message="Solved")

        # initial direction
        d = -g * self.working
        d /= np.linalg.norm(d)

        if self.param['verbose'] >= 10:
            print("%10s %10s %15s %15s %15s" % ("Iteration", "FunEvals",
                                                "Step Length", "Function Val",
                                                "Proj Gradient"))

        k = 0
        while True:
            k += 1

            if self.param['grad_test']:
                self.grad_test(x)

            step_max = self.max_step_size(x, d)
            step_max = min(step_max, self.param['step_max'])
            if self.param['verbose'] >= 100:
                print('lb', self.lb)
                print('x', x)
                print('ub', self.ub)
                print('g', g)
                print('d', d)
                print('step_max', step_max)

            if step_max < 1E-5:
                if self.num_cors() > 0:
                    # maybe clearing up all correction pairs will help
                    if self.param['verbose'] >= 10:
                        print('refresh called')
                    self.init_matrices()

                    # initial direction
                    d = -g * self.working
                    d /= np.linalg.norm(d)
                    continue

            f_old = f
            if self.param['ls'] == 2:
                f, g, x, step, fun_eval_ls = self.line_search(x, d, step_max, f, g, quadratic=True)
            else:
                f, g, x, step, fun_eval_ls = self.line_search(x, d, step_max, f, g, quadratic=False)

            if f > f_old:
                print('error')
                step = None

            if step is None:
                x = x_old
                f, g = self.fg(x)
                fun_eval += 1
                pg = self.proj_grad_norm(x, g)

                self.grad_test(x)
                # line search did not converge
                if self.num_cors() > 0:
                    # maybe clearing up all correction pairs will help
                    if self.param['verbose'] >= 10:
                        print('refresh called')
                    self.init_matrices()

                    # initial direction
                    d = -g * self.working
                    d /= np.linalg.norm(d)
                    continue
                else:
                    # really cannot do any progress due to numerical errors
                    status = 3
                    message = "Line search failed"
                    break

            x = self.force_bounds(x)
            pg = self.proj_grad_norm(x, g)
            fun_eval += fun_eval_ls

            if self.param['verbose'] >= 10:
                print("%10d %10d %15.5g %15.5e %15.5e" % (k, fun_eval, step, f, pg))

            # check for convergence
            if k >= self.param['max_iter']:
                status = 2
                message = "Maximum iterations reached"
                break

            if pg <= self.param['eps_pg']:
                status = 0
                message = "Solved"
                break

            s = x - x_old
            y = g - g_old
            if np.dot(s, y) > eps * np.dot(y, y):
                self.add_corrections(s, y)
            if self.param['verbose'] > 100:
                print(self.S)

            d = -self.two_loop(g)
            dg = np.dot(g, d)
            assert dg < 0
            d = self.project_direction(x, g, d)
            dg = np.dot(g, d)
            assert dg < 0

            x_old = x
            g_old = g
        return OptimizeResult(x=x, fun=f, jac=g,
                              nit=k, nfev=fun_eval,
                              status=status, success=(status==0),
                              message=message)



class Augmented_Lagrangian_NLP:
    def __init__(self, fg, c_f, c_jac, c_lb, c_ub, y, np):
        self.np = np
        self.fg = fg
        self.c_f = c_f
        self.c_jac = c_jac
        self.c_lb = c_lb
        self.c_ub = c_ub
        self.rho = None
        self.y = y

    def constraint_error(self, x):
        c = self.c_f(x)
        cl = c - self.c_lb
        cu = c - self.c_ub
        aug_Lag = self.np.minimum(cl + self.y / self.rho, 0.) + \
                  self.np.maximum(cu + self.y / self.rho, 0.)
        return cl, cu, aug_Lag

    def aug_Lag_fg(self, x):
        f, g = self.fg(x)
        _, _, aug_Lag = self.constraint_error(x)

        f = f + self.rho / 2 * self.np.linalg.norm(aug_Lag) ** 2
        v = self.rho * (aug_Lag)
        c_g = self.c_jac(x, v)

        # No += since we might overwrite the g from genoNLP.
        g = g + c_g
        return f, g


class Augmented_Lagrangian:
    """
    An augmented Lagrangian solver for solving constrained optimization
    problems of the form
    quasi-Newton solver for solving bound-constrained optimization problems
    of the form

        min_x f(x)
        s.t.  cl <= g(x) <= cu
              lb <= x <= ub

    If converts the constrained problem into a sequence of bound-constrained
    problems and solves them using the L-BFGS-B solver.
    """
    def __init__(self, aug_Lag_NLP, x0, np, lb=None, ub=None, options=None):
        if options is None:
            options = {}
        self.NLP = aug_Lag_NLP
        self.x = x0
        self.lb = lb
        self.ub = ub
        self.np = np
        n, = self.NLP.c_lb.shape
        self.y = np.zeros(n)
        self.set_options(options)

    def set_options(self, options):
        all_options = {'verbose', 'max_iter', 'step_max', 'max_ls',
                       'eps_pg', 'm', 'grad_test', 'ls',
                       'max_iter_outer', 'constraint_tol'}
        unsupported = [opt for opt in options.keys() if opt not in all_options]
        for opt in unsupported:
            warnings.warn(f"Option '{opt}' is not supported.", RuntimeWarning)

        self.param = options.copy()
        self.param.setdefault('verbose', 0)
        self.param.setdefault('max_iter_outer', 100)
        self.param.setdefault('constraint_tol', 1E-3)

        self.param_LBFGSB = options.copy()
        self.param_LBFGSB.pop('max_iter_outer', None)
        self.param_LBFGSB.pop('constraint_tol', None)

    def minimize(self):
        np = self.np
        rho = 1.0
        k = 0
        fun_eval = 0
        n_inner = 0
        jac = 0
        constraint_error_old = np.inf
        rho_increases = 0
        while True:
            k += 1
            if self.param['verbose'] >= 5:
                print('outer iteration', k)
                print('rho', rho)
                print('y', self.y)

            self.NLP.rho = rho
            self.NLP.y = self.y
            solver = LBFGSB(self.NLP.aug_Lag_fg, self.x, np, self.lb, self.ub, self.param_LBFGSB)
            res = solver.minimize()

            self.x = res.x
            fun_eval += res.nfev
            n_inner += res.nit
            jac = res.jac

            cl, cu, aug_constraint_error = self.NLP.constraint_error(self.x)
            constraint_error = np.minimum(cl, 0.) + np.maximum(cu, 0.)
            if self.param['verbose'] >= 90:
                print('x', self.x)
                print('c_f', self.NLP.c_f(self.x))
                print('cl', cl)
                print('cu', cu)
            if self.param['verbose'] >= 5:
                print('constraint_error', constraint_error)

            constraint_error_norm = np.linalg.norm(constraint_error, np.inf)
            if constraint_error_norm < self.param['constraint_tol']:
                status = res.status
                message = res.message
                break

            if res.status==1: # augmented Lagrangian could not be solved
                status = 1
                message = "Infeasible"
                break

            if k >= self.param['max_iter_outer']:
                status = 2
                message = "Maximum outer iterations reached"
                break

            self.y = rho * aug_constraint_error
            if constraint_error_norm > constraint_error_old * 0.5:
                rho *= 2
                rho_increases += 1
            else:
                rho_increases = 0
            constraint_error_old = constraint_error_norm

            if rho_increases > 20:
                status = 1
                message = "Infeasible"
                break # problem seems infeasible

        f, _ = self.NLP.fg(self.x)
        return OptimizeResult(x=self.x, y=self.y, fun=f, jac=jac,
                              nit=k, nfev=fun_eval, nInner=n_inner,
                              maxcv=constraint_error_norm,
                              slack=0,
                              status=status, success=(status==0),
                              message=message)



def minimize(fg, x0, lb=None, ub=None, options=None, constraints=None, np=None):
    if np is None:
        import numpy as np
    if options is None:
        options = {}
    x0 = np.ascontiguousarray(np.array(x0))
    if not lb is None:
        lb = np.ascontiguousarray(np.array(lb))
    if not ub is None:
        ub = np.ascontiguousarray(np.array(ub))
    if not constraints:
        solver = LBFGSB(fg, x0, np, lb, ub, options)
    else:
        if isinstance(constraints, dict):
            constraints = (constraints, )

        shape_constraints = []
        offset = [0]
        c_lb_all = []
        c_ub_all = []
        for c in constraints:
            # determine shape of constraint i and its length
            dummy_f_c = c['fun'](x0)
            shape_constraints.append(dummy_f_c.shape)
            m = len(dummy_f_c.reshape(-1))
            offset.append(offset[-1] + m)

            # check the type of constraints
            if c['type'] == 'eq':
                c_lb = np.zeros(m)
                c_ub = np.zeros(m)
            elif c['type'] == 'ineq':
                c_lb = np.full(m, -np.inf)
                c_ub = np.zeros(m)
            else:
                assert False

            c_lb_all.append(c_lb)
            c_ub_all.append(c_ub)

        mTotal = offset[-1]
        c_lb_all = np.concatenate(c_lb_all)
        c_ub_all = np.concatenate(c_ub_all)

        def c_f_all(x):
            l = [c['fun'](x).reshape(-1) for c in constraints]
            f = np.concatenate(l)
            return f
        def c_jac_all(x, v):
            g = np.zeros_like(x)
            for i, c in enumerate(constraints):
                g = g + c['jacprod'](x, v[offset[i]:offset[i+1]].reshape(shape_constraints[i])).reshape(-1)
            return g

        y = np.zeros(mTotal)

        augmented_Lagrangian_NLP = Augmented_Lagrangian_NLP(fg, c_f_all, c_jac_all,
                                                            c_lb_all, c_ub_all, y, np)
        solver = Augmented_Lagrangian(augmented_Lagrangian_NLP, x0, np, lb, ub, options)
    return solver.minimize()

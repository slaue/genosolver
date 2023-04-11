# -*- coding: utf-8 -*-

#    GENO is a solver for non-linear optimization problems.
#    It can solve constrained and unconstrained problems.
#
#    Copyright (C) 2021-2022 Soeren Laue
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Affero General Public License as published
#    by the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Affero General Public License for more details.
#
#    You should have received a copy of the GNU Affero General Public License
#    along with this program. If not, see <http://www.gnu.org/licenses/>.
#
#    Contact the developer:
#
#    E-mail: soeren.laue@uni-jena.de
#    Web:    http://www.geno-project.org

import sys, os

HERE_TEST_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE_TEST_PATH, '../'))

from genosolver import minimize, check_version

import numpy as np
from math import inf
from numpy.testing import assert_allclose
import pytest


def test_version():
    check_version('0.0.1')
    check_version('1.0.0')

def test_unconstrained0():
    def fg(x):
        f = x ** 2
        g = 2 * x
        return f, g

    res = minimize(fg, 100., np=np)
    print(res)
    assert_allclose(res.x, [0], atol=1E-4)

def test_unconstrained1():
    def fg(x):
        f = x.dot(x) + np.sum(x)
        g = 2*x + np.ones_like(x)
        return f, g

    x0 = np.ones(5)
    xOpt = -0.5*x0
    res = minimize(fg, x0, np=np)
    assert_allclose(res.x, xOpt, atol=1E-4)

def test_unconstrained2():
    def fg(x):
        n = len(x)
        c = np.arange(n)
        f = x.dot(x) + c.dot(x)
        g = 2*x + c
        return f, g

    n = 5
    x0 = np.ones(n)
    c = np.arange(n)
    xOpt = -0.5*c
    res = minimize(fg, x0, np=np)
    assert_allclose(res.x, xOpt, atol=1E-4)

def test_bound_constrained0():
    def fg(x):
        n = len(x)
        c = np.arange(n)
        f = x.dot(x) + c.dot(x)
        g = 2*x + c
        return f, g

    n = 5
    x0 = np.ones(n)
    c = np.arange(n)
    n2 = n//2
    bnds = [(0, 1)] * n2
    bnds += [(-inf, 1)] * (n-n2)
    lb = [0] * n2 + [-inf] * (n-n2)
    ub = [1] * n
    xOpt = -0.5*c
    xOpt[:n2] = 0
    res = minimize(fg, x0, lb=lb, ub=ub, np=np)
    assert_allclose(res.x, xOpt, atol=1E-4)


def test_bound_constrained1():
    def fg1(x):
        Q = np.array([[1., 0, 1],
                      [0, 2, 0],
                      [1, 0, 3]])
        c = np.array([2., 3, 4])
        t = Q.dot(x)
        f = 0.5 * x.dot(t) + c.dot(x)
        g = t + c
        return f, g

    options = {'verbose' : 100,
               'eps_pg' : 1E-6,
               'm' : 10,
               'max_iter' : 1000,
               'grad_test' : False,
               'ls' : 0}

    x0 = np.zeros(3)
    res = minimize(fg1, x0, options=options, np=np)
    assert_allclose(res.x, np.array([-1, -1.5, -1]), atol=1E-4)

    x0 = np.zeros(3)
    lb = np.full(3, -1.)
    ub = np.full(3, 5)
    res = minimize(fg1, x0, lb=lb, ub=ub, np=np)
    assert_allclose(res.x, np.array([-1, -1, -1]), atol=1E-4)

    lb = np.full(3, -1.2)
    ub = np.full(3, 5)
    res = minimize(fg1, x0, lb=lb, ub=ub, np=np)
    assert_allclose(res.x, np.array([-1, -1.2, -1]), atol=1E-4)

    lb = np.full(3, -2)
    ub = np.full(3, 5)
    res = minimize(fg1, x0, lb=lb, ub=ub, np=np)
    assert_allclose(res.x, np.array([-1, -1.5, -1]), atol=1E-4)




def test_constrained0():
    def fg(x):
        n = len(x)
        c = np.arange(n)
        f = x.dot(x) + c.dot(x)
        g = 2*x + c
        return f, g

    def constraint_f(x):
        # sum(x) - 1 == 0
        f = np.sum(x) - 1
        return f

    def constraint_jac_prod(x, y):
        # sum(x) - 1 == 0
        # return the product of the Jacobian and y
        g = np.ones_like(x)
        jp = y * g
        return jp

    constraints = ({'type' : 'eq',
                    'fun' : constraint_f,
                    'jacprod' : constraint_jac_prod})

    options = {'eps_pg' : 1E-4,
               'constraint_tol' : 1E-4,
               'max_iter' : 3000,
               'm' : 10,
               'ls' : 0,
               'verbose' : 10  # Set it to 0 to fully mute it.
              }
    n = 4
    x0 = np.zeros(n)
    res = minimize(fg, x0, constraints=constraints, options=options, np=np)
    assert_allclose(res.x, np.array([1., 0.5, 0, -0.5]), atol=1E-4)

    lb = np.full(n, 0)
    ub = np.full(n, 1)
    res = minimize(fg, x0, lb=lb, ub=ub, constraints=constraints, options=options, np=np)
    assert_allclose(res.x, np.array([0.75, 0.25, 0, 0]), atol=1E-4)

    lb = np.full(n, -0.2)
    res = minimize(fg, x0, lb=lb, constraints=constraints, options=options, np=np)
    assert_allclose(res.x, np.array([0.9, 0.4, -0.1, -0.2]), atol=1E-4)

    ub = np.full(n, 0.5)
    res = minimize(fg, x0, ub=ub, constraints=constraints, options=options, np=np)
    assert_allclose(res.x, np.array([0.5, 0.5, 0.25, -0.25]), atol=1E-4)


def test_infeasible_bounds():
    def fg(x):
        return 0, np.zeros(4)

    n = 4
    x0 = np.ones(n)
    lb = np.full(n, 1)
    ub = np.full(n, 0)
    res = minimize(fg, x0, lb, ub, np=np)
    assert not res.success



def test_infeasible_constrained():
    def fg(x):
        n = len(x)
        c = np.arange(n)
        f = x.dot(x) + c.dot(x)
        g = 2*x + c
        return f, g

    def constraint_f(x):
        # sum(x) - 1 == 0
        f = np.sum(x) - 1
        return f

    def constraint_jac_prod(x, y):
        # sum(x) - 1 == 0
        # return the product of the Jacobian and y
        g = np.ones_like(x)
        jp = y * g
        return jp

    constraints = ({'type' : 'eq',
                    'fun' : constraint_f,
                    'jacprod' : constraint_jac_prod})

    n = 4
    x0 = np.ones(n)
    lb = np.full(n, 0)
    ub = np.full(n, 0.2)
    res = minimize(fg, x0, lb=lb, ub=ub, constraints=constraints, np=np)
    assert not res.success


def test_infeasible_bounds_constrained():
    def fg(x):
        n = len(x)
        c = np.arange(n)
        f = x.dot(x) + c.dot(x)
        g = 2*x + c
        return f, g

    def constraint_f(x):
        # sum(x) - 1 == 0
        f = np.sum(x) - 1
        return f

    def constraint_jac_prod(x, y):
        # sum(x) - 1 == 0
        # return the product of the Jacobian and y
        g = np.ones_like(x)
        jp = y * g
        return jp

    constraints = ({'type' : 'eq',
                    'fun' : constraint_f,
                    'jacprod' : constraint_jac_prod})

    n = 4
    x0 = np.ones(n)
    lb = np.full(n, 1)
    ub = np.full(n, 0)
    res = minimize(fg, x0, lb=lb, ub=ub, constraints=constraints, np=np)
    assert not res.success


def test_iterations_max():
    def fg(x):
        n = len(x)
        c = np.arange(n)
        f = x.dot(x) + c.dot(x)
        g = 2*x + c
        return f, g

    options = {'max_iter' : 1}

    x0 = np.arange(5)
    res = minimize(fg, x0, options=options, np=np)
    print(res)
    assert not res.success
    assert res.status==2

def test_iterations_max_constrained():
    def fg(x):
        n = len(x)
        c = np.arange(n)
        f = x.dot(x) + c.dot(x)
        g = 2*x + c
        return f, g

    def constraint_f(x):
        # sum(x) - 1 == 0
        f = np.sum(x) - 1
        return f

    def constraint_jac_prod(x, y):
        # sum(x) - 1 == 0
        # return the product of the Jacobian and y
        g = np.ones_like(x)
        jp = y * g
        return jp

    constraints = ({'type' : 'eq',
                    'fun' : constraint_f,
                    'jacprod' : constraint_jac_prod})

    options = {'eps_pg' : 1E-4,
               'constraint_tol' : 1E-4,
               'max_iter' : 1,
               'm' : 10,
               'ls' : 0,
               'verbose' : 0  # Set it to 0 to fully mute it.
              }

    n = 4
    x0 = np.zeros(n)
    res = minimize(fg, x0, constraints=constraints, options=options, np=np)
    assert res.status==2



def test_outer_iterations_max_constrained():
    def fg(x):
        n = len(x)
        c = np.arange(n)
        f = x.dot(x) + c.dot(x)
        g = 2*x + c
        return f, g

    def constraint_f(x):
        # sum(x) - 1 == 0
        f = np.sum(x) - 1
        return f

    def constraint_jac_prod(x, y):
        # sum(x) - 1 == 0
        # return the product of the Jacobian and y
        g = np.ones_like(x)
        jp = y * g
        return jp

    constraints = ({'type' : 'eq',
                    'fun' : constraint_f,
                    'jacprod' : constraint_jac_prod})

    options = {'eps_pg' : 1E-4,
               'constraint_tol' : 1E-4,
               'max_iter_outer' : 1,
               'm' : 10,
               'ls' : 0,
               'verbose' : 0  # Set it to 0 to fully mute it.
              }

    n = 4
    x0 = np.zeros(n)
    res = minimize(fg, x0, constraints=constraints, options=options, np=np)
    assert res.status == 2


def test_unconstrained_wrong_ls():
    def fg(x):
        n = len(x)
        c = np.arange(n) + 1
        f = np.sum(c*x**4) + np.sum(x)
        g = 4*c*x**3 + np.ones(n)
        return f, g

    options = {'ls' : 2,
               'verbose' : 10}

    x0 = np.ones(5)
    c = np.arange(5) + 1
    res = minimize(fg, x0, options=options, np=np)
    assert_allclose(res.x, -1/(4*c)**(1/3), atol=1E-4)

def test_unconstrained_quadratic_ls():
    def fg(x):
        n = len(x)
        c = np.arange(n) + 1
        f = np.sum(c*x**2) + np.sum(x)
        g = 2*x*c + np.ones(n)
        return f, g

    options = {'ls' : 2,
               'verbose' : 0}

    x0 = np.ones(50)
    c = np.arange(50) + 1
    res = minimize(fg, x0, options=options, np=np)
    assert_allclose(res.x, -1/(2*c), atol=1E-4)


def test_grad_test():
    def fg(x):
        n = len(x)
        c = np.arange(n) + 1
        f = np.sum(c*x**2) + np.sum(x)
        g = 2*x*c + np.ones(n)
        return f, g

    options = {'ls' : 0,
               'verbose' : 10,
               'grad_test' : True}

    x0 = np.ones(5)
    c = np.arange(5) + 1
    res = minimize(fg, x0, options=options, np=np)
    assert_allclose(res.x, -1/(2*c), atol=1E-4)

def test_wrong_grad():
    def fg(x):
        n = len(x)
        c = np.arange(n) + 1
        f = np.sum(c*x**2) + np.sum(x)
        g = x*c + np.ones(n) # wrong gradient, should be 2*x*c + 1
        return f, g

    options = {'ls' : 0,
               'verbose' : 10}

    x0 = np.ones(5)
    res = minimize(fg, x0, options=options, np=np)
    print(res)
    # line search should fail due to wrong gradient
    assert res.status==3

def test_brownbs():
    x0 = np.ones(2)
    def f(x):
        return (x[0] - 1e6)**2 + (x[1] - 2e-6) ** 2 + (x[0] * x[1] - 2.) ** 2

    def g(x):
        xd0 = 2 * (x[0] - 1e6) + 2 * (x[0] * x[1] - 2.) * x[1]
        xd1 = 2 * (x[1] - 2e-6) + 2 * (x[0] * x[1] - 2.) * x[0]
        return np.array([ xd0, xd1 ])
    
    #g = grad(f)
    fg = lambda x: (f(x), g(x))

    np.seterr(all = 'raise')
    options = { 'ls': 0, 'verbose': 101, 'max_iter': 20 }
    res = minimize(fg, x0, options = options, np = np)
    print(res)
    assert res.status == 0


'''
from autograd import grad
import autograd.numpy as anp

def test_fletchbv():
    x0 = anp.array([ (i + 1) / 1001 for i in range(1000) ])
    def f(x):
        return 0.5 * x[0] ** 2 + 0.5 * anp.sum((x[:-1] - x[1:]) ** 2) + 0.5 * x[-1] ** 2 - (1 + 2 / 1001 ** 2) * anp.sum(x) - anp.sum(anp.cos(x) / 1001 ** 2)

    g = grad(f)
    fg = lambda x: (f(x), g(x))

    np.seterr(all = 'raise')
    options = { 'ls': 0, 'verbose': 0, 'max_iter': 400 }
    res = minimize(fg, x0, options = options, np = np)
    print(res)
    assert res.status == 0
'''

if __name__ == "__main__":
    pytest.main()

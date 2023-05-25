# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 15:33:07 2023

@author: Sören
"""

import numpy as np

from genosolver import minimize
from scipy.optimize import minimize as s_min
from numpy.testing import assert_allclose

def test_bound_constrained0():
    def fg1(x):
        c = np.array([2.])
        f = c.dot(x)
        g = c
        return f, g

    options = {'verbose' : 10,
               'eps_pg' : 1E-6,
               'm' : 10,
               'max_iter' : 1000,
               'grad_test' : False,
               'ls' : 0}

#    x0 = np.zeros(3)
#    res = minimize(fg1, x0, options=options, np=np)
#    assert_allclose(res.x, np.array([-1, -1.5, -1]), atol=1E-4)

    x0 = np.zeros(1)
    lb = np.full(1, -1.)
    ub = np.full(1, 5)
    res = minimize(fg1, x0, lb=lb, ub=ub, np=np, options=options)
    assert_allclose(res.x, np.array([-1]), atol=1E-4)



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

    options = {'verbose' : 10,
               'eps_pg' : 1E-6,
               'm' : 10,
               'max_iter' : 1000,
               'grad_test' : False,
               'ls' : 0}

#    x0 = np.zeros(3)
#    res = minimize(fg1, x0, options=options, np=np)
#    assert_allclose(res.x, np.array([-1, -1.5, -1]), atol=1E-4)

    x0 = np.zeros(3)
    lb = np.full(3, -1.)
    ub = np.full(3, 5)
    res = minimize(fg1, x0, lb=lb, ub=ub, np=np, options=options)
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
#    res = minimize(fg, x0, constraints=constraints, options=options, np=np)
#    assert_allclose(res.x, np.array([1., 0.5, 0, -0.5]), atol=1E-4)

    lb = np.full(n, 0)
    ub = np.full(n, 1)
#    res = minimize(fg, x0, lb=lb, ub=ub, constraints=constraints, options=options, np=np)
#    assert_allclose(res.x, np.array([0.75, 0.25, 0, 0]), atol=1E-4)

    lb = np.full(n, -0.2)
    res = minimize(fg, x0, lb=lb, constraints=constraints, options=options, np=np)
    assert_allclose(res.x, np.array([0.9, 0.4, -0.1, -0.2]), atol=1E-4)

    ub = np.full(n, 0.5)
#    res = minimize(fg, x0, ub=ub, constraints=constraints, options=options, np=np)
#    assert_allclose(res.x, np.array([0.5, 0.5, 0.25, -0.25]), atol=1E-4)


def test_brownbs():
    x0 = np.ones(2)
    def f(x):
        return (x[0] - 1e6)**2 + (x[1] - 2e-6) ** 2 + (x[0] * x[1] - 2.) ** 2

    def g(x):
        xd0 = 2 * (x[0] - 1e6) + 2 * (x[0] * x[1] - 2.) * x[1]
        xd1 = 2 * (x[1] - 2e-6) + 2 * (x[0] * x[1] - 2.) * x[0]
        return np.array([ xd0, xd1 ])
    
    fg = lambda x: (f(x), g(x))

    np.seterr(all='raise')
    options = { 'ls': 0, 'verbose': 10, 'max_iter': 20 }
    res = minimize(fg, x0, options=options, np=np)
    print(res)
    
    print('*'*20)
    options = {'disp' : True}
    res = s_min(fg, x0, jac=True, options=options, method='BFGS')
    print(res)
    assert res.status == 0


if __name__ == '__main__':
#    test_brownbs()
#    test_bound_constrained1()
#    test_constrained0()
    test_bound_constrained0()

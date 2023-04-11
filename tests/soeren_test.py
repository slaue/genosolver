# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 15:33:07 2023

@author: Sören
"""

import numpy as np

from genosolver import minimize
from scipy.optimize import minimize as s_min


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
#    options = { 'ls': 0, 'verbose': 10, 'max_iter': 20 }
#    res = minimize(fg, x0, options = options, np = np)
    options = {'disp' : True}
    res = s_min(fg, x0, jac=True, options=options, method='BFGS')
    print(res)
    assert res.status == 0


if __name__ == '__main__':
    test_brownbs()

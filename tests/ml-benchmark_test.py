import sys, os

HERE_TEST_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE_TEST_PATH, '../'))

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
    options = { 'ls': 0, 'verbose': 0, 'max_iter': 30 }
    res = minimize(fg, x0, options = options, np = np)
    print(res)
    assert res.status == 0


from autograd import grad
import autograd.numpy as anp

def test_fletchbv():
    x0 = anp.array([ (i + 1) / 1001 for i in range(1000) ])
    def f(x):
        return 0.5 * x[0] ** 2 + 0.5 * anp.sum((x[:-1] - x[1:]) ** 2) + 0.5 * x[-1] ** 2 - (1 + 2 / 1001 ** 2) * anp.sum(x) - anp.sum(anp.cos(x) / 1001 ** 2)

    g = grad(f)
    fg = lambda x: (f(x), g(x))

    np.seterr(all = 'raise')
    options = { 'ls': 0, 'verbose': 0, 'max_iter': 400, 'm': 30 }
    res = minimize(fg, x0, options = options, np = np)
    #print(res)
    assert res.status == 0

if __name__ == '__main__':
    test_brownbs()
    test_fletchbv()
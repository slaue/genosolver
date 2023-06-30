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
    options = { 'ls': 0, 'verbose': 10, 'max_iter': 30 }
    res = minimize(fg, x0, options = options, np = np)
    print(res)
    assert res.status == 0

import warnings

def test_fletchbv():
    try:
        from autograd import grad
        import autograd.numpy as anp
    except ImportError as e:
        warnings.warn(f'test_fletchbv not executed\n{e}')
        return
    
    x0 = anp.array([ (i + 1) / 1001 for i in range(1000) ])
    def f(x):
        return 0.5 * x[0] ** 2 + 0.5 * anp.sum((x[:-1] - x[1:]) ** 2) + 0.5 * x[-1] ** 2 - (1 + 2 / 1001 ** 2) * anp.sum(x) - anp.sum(anp.cos(x) / 1001 ** 2)

    g = grad(f)
    fg = lambda x: (f(x), g(x))

    np.seterr(all='raise')
    options = { 'ls': 0, 'verbose': 0, 'max_iter': 1000, 'm': 30 }
    res = minimize(fg, x0, options = options, np = np)
    #print(res)
    assert res.status == 0

def test_cliff():
    x0 = np.array([0., -1.])
    def f(x):
        return (0.01 * x[0] - 0.03) ** 2 - x[0] + x[1] + np.exp(20 * (x[0] - x[1]))
    def g(x):
        x0 = 0.02 * (0.01 * x[0] - 0.03) - 1 + 20 * np.exp(20 * (x[0] - x[1]))
        x1 = 1 - 20 * np.exp(20. * (x[0] - x[1]))
        return np.array([x0,x1])
    def fg(x):
        return (f(x),g(x))
    
    options = { 'ls': 0, 'verbose': 0, 'max_iter': 60 }
    res = minimize(fg, x0, options=options, np=np)
    assert res.status == 0

def test_qr3dbd():
    try:
        from autograd import grad
        import autograd.numpy as anp
    except ImportError as e:
        warnings.warn(f'test_qr3dbd not executed\n{e}')
        return

    m = 10
    A = anp.zeros((m, m))
    for i in range(m):
        if i < m - 1:
            A[i+1, i] = (-i-1) / m
            if i > 0:
                A[i, i+1] = (-i) / m
        A[i, i] = 2 * (i + 1) / m
    A[m-1, m-1] *= 10
    I = anp.eye(m)

    # Variables:
    Q = anp.eye(m)
    Q_lower = anp.ones((m,m)) * -anp.inf
    Q_upper = anp.ones((m,m)) * anp.inf

    R = anp.triu(A)
    R_lower = anp.triu(np.ones((m,m)) * -anp.inf)
    R_upper = anp.triu(np.ones((m,m)) * anp.inf)
    for i in range(m):
        R_lower[i, i] = 0
        for j in range(m):
            if j > i + 2:
                R_lower[i, j] = R_upper[i, j] = 0

    def f(QR):
        Q = QR[:m*m].reshape(m,m)
        R = QR[m*m:].reshape(m,m)
        val = anp.sum((Q @ R - A) ** 2)
        val2 = anp.sum(anp.triu(Q @ Q.T - I) ** 2)
        return val + val2

    x0 = anp.block([ Q.reshape(-1), R.reshape(-1) ])

    g = grad(f)
    fg = lambda x: (f(x), g(x))

    np.seterr(all='raise')
    options = { 'ls': 0, 'verbose': 0, 'max_iter': 1000 }
    res = minimize(fg, x0, options=options, np=np)

    assert res.status == 0
    

def test_qr3dls():
    try:
        from autograd import grad
        import autograd.numpy as anp
    except ImportError as e:
        warnings.warn(f'test_qr3dls not executed\n{e}')
        return

    m = 10
    A = np.zeros((m, m))
    for i in range(m):
            if i < m - 1:
                    A[i + 1, i] = (-i - 1) / m
                    if i > 0:
                            A[i, i + 1] = (-i) / m
            A[i, i] = 2 * (i + 1) / m
    A[m - 1, m - 1] *= 10
    I = np.eye(m)

    # Variables:
    Q = np.eye(m)
    Q_lower = np.ones((m, m)) * -np.inf
    Q_upper = np.ones((m, m)) * np.inf

    R = np.triu(A)
    R_lower = np.triu(np.ones((m, m)) * -np.inf)
    R_upper = np.triu(np.ones((m, m)) * np.inf)
    for i in range(m):
            R_lower[i, i] = 0

    
    def f(QR):
        Q = QR[:m*m].reshape(m,m)
        R = QR[m*m:].reshape(m,m)
        val = anp.sum((Q @ R - A) ** 2)
        val2 = anp.sum(anp.triu(Q @ Q.T - I) ** 2)
        return val + val2

    x0 = anp.block([ Q.reshape(-1), R.reshape(-1) ])

    g = grad(f)
    fg = lambda x: (f(x), g(x))

    np.seterr(all='raise')
    options = { 'ls': 0, 'verbose': 0, 'max_iter': 1000 }
    res = minimize(fg, x0, options=options, np=np)

    assert res.status == 0		

def test_qr3d():
    try:
        from autograd import grad
        import autograd.numpy as anp
    except ImportError as e:
        warnings.warn(f'test_qr3d not executed\n{e}')
        return

    m = 10
    A = np.zeros((m, m))
    for i in range(m):
            if i < m - 1:
                    A[i + 1, i] = (-i - 1) / m
                    if i > 0:
                            A[i, i + 1] = (-i) / m
            A[i, i] = 2 * (i + 1) / m
    A[m - 1, m - 1] *= 10
    I = np.eye(m)

    # Variables:
    Q = np.eye(m)
    Q_lower = np.ones((m, m)) * -np.inf
    Q_upper = np.ones((m, m)) * np.inf

    R = np.triu(A)
    R_lower = np.triu(np.ones((m, m)) * -np.inf)
    R_upper = np.triu(np.ones((m, m)) * np.inf)
    for i in range(m):
            R_lower[i, i] = 0

    
    def f(QR):
        Q = QR[:m*m].reshape(m,m)
        R = QR[m*m:].reshape(m,m)
        val = anp.sum((Q @ R - A) ** 2)
        val2 = anp.sum(anp.triu(Q @ Q.T - I) ** 2)
        return val + val2

    x0 = anp.block([ Q.reshape(-1), R.reshape(-1) ])

    g = grad(f)
    fg = lambda x: (f(x), g(x))

    np.seterr(all='raise')
    options = { 'ls': 0, 'verbose': 0, 'max_iter': 1000 }
    res = minimize(fg, x0, options=options, np=np)

    assert res.status == 0


def test_explin2():
    try:
        from autograd import grad
        import autograd.numpy as anp
    except ImportError as e:
        warnings.warn(f'test_explin2 not executed\n{e}')
        return

    n = 120
    m = 10

    # Variables:
    x_offset = 0
    x_size = n
    x = anp.ones(x_size)*0.0
    x_lower = np.zeros(x_size)
    x_upper = np.ones(x_size)*10.0

    def f(x):
        return anp.sum(anp.exp(0.1 * (anp.arange(m) + 1) * x[:m] * x[1:m+1] / m)) + anp.sum(-10.0 * (anp.arange(n) + 1) * x)
    g = grad(f)
    fg = lambda x: (f(x), g(x))

    np.seterr(all='raise')
    options = { 'ls': 0, 'verbose': 0, 'max_iter': 1000 }
    res = minimize(fg, x, options=options, np=np)

    assert res.status == 0


def test_explin():
    try:
        from autograd import grad
        import autograd.numpy as anp
    except ImportError as e:
        warnings.warn(f'test_explin not executed\n{e}')
        return

    n = 120
    m = 10

    # Variables:
    x_offset = 0
    x_size = n
    x = np.ones(x_size)*0.0
    x_lower = np.zeros(x_size)
    x_upper = np.ones(x_size)*10.0

    def f(x):
        return anp.sum(anp.exp(0.1 * x[:m] * x[1:m+1])) + anp.sum(-10 * (np.arange(n) + 1) * x)

    g = grad(f)
    fg = lambda x: (f(x), g(x))

    np.seterr(all='raise')
    options = { 'ls': 0, 'verbose': 0, 'max_iter': 1000 }
    res = minimize(fg, x, options=options, np=np)

    assert res.status == 0


def test_cvxbqp1():
    try:
        from autograd import grad
        import autograd.numpy as anp
    except ImportError as e:
        warnings.warn(f'test_cvxbqp1 not executed\n{e}')
        return

    N = 10000

    # Variables:
    x_offset = 0
    x_size = N
    x = np.ones(x_size)*0.5
    x_lower = np.ones(x_size)*0.1
    x_upper = np.ones(x_size)*10.0


    def f(x):
        rng = anp.arange(N)
        return anp.sum(0.5 * anp.arange(1, N + 1) * anp.square(x + x[(2 * rng + 1) % N] + x[(3 * rng + 2) % N]))
    # return np.sum([0.5 * (i + 1) * (x[i] + x[(2 * (i + 1) - 1) % N] + x[(3 * (i + 1) - 1) % N]) ** 2 for i in range(N)])

    g = grad(f)
    fg = lambda x: (f(x), g(x))

    np.seterr(all='raise')
    options = { 'ls': 0, 'verbose': 0, 'max_iter': 1000 }
    res = minimize(fg, x, options=options, np=np)

    assert res.status == 0


        
if __name__ == '__main__':
    test_brownbs()
    test_fletchbv()
    test_cliff()
    test_qr3dbd()
    test_qr3dls()
    test_qr3d()
    test_explin()
    test_explin2()
    test_cvxbqp1()

import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt


#parameters
N = 1000
M = 25
P = 2 * M
sigma = 0.04
PI = np.pi
C = np.diag(1 / np.arange(1, P + 1) ** 2)

#data
x = [0.2, 0.4, 0.4, 0.8]
y = [0.5041, 0.8505, 1.2257, 1.41113]

xi = np.zeros(P)


def S(x, xi):
    '''
    compute integral
    '''
    k = np.linspace(1, P, P, True)
    # define f^(-u(x, xi))
    f = lambda u: np.exp(-np.sqrt(2) / PI * np.sum(xi * np.sin(k * PI * u)))

    # discretization of interval [0, x]
    y = np.linspace(0, x, M)

    val = np.zeros(M)
    for i, y_i in enumerate(y):
        val[i] = f(y_i)

    return integrate.trapz(val, y)

def G(xi):
    '''
    compute G(xi)
    '''
    g = []
    # compute S_1
    S1 = S(1, xi)
    for i in range(4):
        # compute pressure p(x, xi)
        p = 2 * S((i + 1) * 0.2, xi) / S1
        g.append(p)
    return np.asarray(g)

def pi_exp(xi):
    '''
    likelihood pi(y, xi)
    '''
    diff = y - G(xi)
    diff = diff[:, np.newaxis]
    return - diff.T @ diff / 2 / sigma**2

def pi_0(xi):
    '''
    compute the prior pi_0
    '''
    # 1..P
    enum = np.linspace(1, P, P, True)
    # exponent
    exp = np.sum(xi ** 2 * enum ** 2 / 2)
    # print(f'exp {exp}')
    return - exp

def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[len(x) - 1 :]

def plots(X):
    f = np.zeros(N)
    for i in range(N):
        f[i] = S(1, X[i])
    ac = autocorr(f)
    print (f'plot')
    
    plt.figure()
    plt.plot(range(N), f)
    plt.show()

def ESS(X):
    """
    compute ESS
    """
    f = lambda xi: S(1, xi)

    for i in range(N):
        sum += corr(f(X[0]), f(X[i]))

    return N * (1 + 2 * sum) ** (-1)

def RWMH(s, improvement = None):
    '''
    Random walk MH algorithm
    '''
    X = np.zeros((N,P))
    X[0] = np.random.random_sample(P)
    if improvement == None:
        mean = lambda u: X[u]
        cov = s**2 * C
    elif improvement == "pCN":
        mean = lambda u: X[u]
        cov = s ** 2 * C

    avg_ap = 0
    for n in range(N - 1):
        proposal = np.random.multivariate_normal(mean(n), cov)
        accept_prob = min(1, np.exp(pi_exp(proposal) - pi_exp(X[n])))
        # print(f'iter {n} : {pi_exp(proposal)} {pi_exp(X[n])}')
        avg_ap += accept_prob
        if accept_prob > np.random.random(1):
            X[n + 1] = proposal
        else:
            X[n + 1] = X[n]

    print(f'avg accept prob {avg_ap / N}')
    return X

if __name__ == '__main__':
    s = 0.5

    X = RWMH(s)
    plots(X)
import numpy as np
import scipy.integrate as integrate
import matplotlib as plt


#parameters
N = 10000
M = 5
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
    f = lambda u: np.exp(-np.sqrt(2) / PI * np.sum(xi* np.sin(u* PI * k)))
    y = np.linspace(0, x, M)
    val = np.zeros(M)
    for i, y_i in enumerate(y):
        val[i] = f(y_i)

    return integrate.trapz(val, y)

def p(x, xi):
    '''
    compute pressure p(x, xi)
    '''
    return 2 * S(x, xi) / S(1, xi)

def G(xi):
    '''
    compute G(xi)
    '''
    g = []
    for i in range(4):
        g.append(p(x[i], xi))
    return np.asarray(g)

def pi_exp(xi):
    '''
    likelihood pi(y, xi)
    '''
    diff = y - G(xi)
    diff = diff[:, np.newaxis]
    return -diff.T @ diff / 2 / sigma**2

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
    return result[result.size/2:]

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

    for n in range(N-1):
        proposal = np.random.normal(mean(n), cov)

        accept_prob = min(1, np.exp(pi_exp(proposal) + pi_0(proposal) - pi_exp(X[n]) - pi_0(X[n])))
        if accept_prob > np.random.random(1):
            X[n+1] = proposal
        else:
            X[n + 1] = X[n]

    return X

if __name__ == '__main__':
    s = 0.5

    X = RWMH(s)
    print(X[N-1])
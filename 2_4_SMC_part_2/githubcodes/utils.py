
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np


def resampling(z):
    N = len(z)
    u = np.random.uniform(size=N)
    qc = np.cumsum(z)
    qc = qc / qc[N - 1]
    u_qc = np.concatenate((u, qc)).reshape(1, -1)
    # print(u_qc)
    ind1 = np.argsort(u_qc)
    ind2 = np.where(ind1 < N)
    ind2 = np.sort(ind2)
    index = ind2[1] - range(N)
    return index




def stateTransFunc(xt, time_t):
    time_t += 1
    return 0.5 * xt + 25 * xt / (1 + xt ** 2) + 8 * np.cos(1.2 * time_t)


def transferFunc(x):
    return x ** 2 / 20


def generateData(f=stateTransFunc, g=transferFunc, Q=1, R=1, x0=0, T=100):
    x = np.zeros(T)
    y = np.zeros(T)
    q = np.sqrt(Q)
    r = np.sqrt(R)
    x[0] = x0  # deterministic initial state
    y[0] = g(x[0]) + r * np.random.normal(size=1)

    for t in range(1, T):
        x[t] = f(x[t - 1], t - 1) + q * np.random.normal(size=1)
        y[t] = g(x[t]) + r * np.random.normal(size=1)

    return x, y


def multinomial_resampling(ws, size=0):
    # Determine number of elements
    if size == 0:
        N = len(ws)
    else:
        N = size  # len(ws)
    # Create a sample of uniform random numbers
    u_sample = stats.uniform.rvs(size=N)
    # Transform them appropriately
    u = np.zeros((N,))
    u[N - 1] = np.power(u_sample[N - 1], 1 / N)
    for i in range(N - 1, 0, -1):
        u[i - 1] = u[i] * np.power(u_sample[i - 1], 1 / i)

    # Output array
    out = np.zeros((N,), dtype=int)

    # Find the right ranges
    total = 0.0
    i = 0
    j = 0
    while j < N and i < N:
        total += ws[i]
        while j < N and total > u[j]:
            out[j] = i
            j += 1

        # Increase weight counter
        i += 1

    return out


def systematic_resampling(ws, size=0):
    # Determine number of elements
    if size == 0:
        N = len(ws)
    else:
        N = size  # len(ws)
    # Output array
    out = np.zeros((N,), dtype=int)

    # Create one single uniformly distributed number
    u = stats.uniform.rvs() / N
    # Find the right ranges
    total = ws[0]
    j = 0
    for i in range(N):
        while total < u:
            j += 1
            total += ws[j]

        # Once the right index is found, save it
        out[i] = j
        u = u + 1 / N

    return out


def stratified_resampling(ws, size=0):
    # Determine number of elements
    if size == 0:
        N = len(ws)
    else:
        N = size  # len(ws)
    # Output array
    out = np.zeros((N,), dtype=int)

    # Find the right ranges
    total = ws[0]
    j = 0
    for i in range(N):
        u = (stats.uniform.rvs() + i) / N
        while total < u:
            j += 1
            total += ws[j]

        # Once the right index is found, save it
        out[i] = j

    return out


# plot geneology of the survived particles
def particleGeneology(particles, B, lengthGeneology=10):
    N, T = particles.shape
    x_matrix = np.zeros((N, T))
    startIndex = T - lengthGeneology

    for t in range(T):
        x_matrix[:, t] = t

    # plot all the particles first
    plt.scatter(x_matrix[:, startIndex:T], particles[:, startIndex:T], s=10)

    # plot geneology
    x_star = np.zeros(lengthGeneology)
    for j in range(N):
        index = 0
        for t in range(startIndex, T):
            x_star[index] = particles[B[j, t], t]
            index = index + 1

        x_dim = list(range(startIndex, T))
        plt.plot(x_dim, x_star, lw=1, color='grey')
    plt.show()


# plot geneology of all the particles generated (survived and died)
def particleGeneologyAll(particles, B, lengthGeneology=10):
    N, T = particles.shape
    x_matrix = np.zeros((N, T))
    startIndex = T - lengthGeneology

    for t in range(startIndex, T):
        x_matrix[:, t] = t

    # plot all the particles first
    plt.scatter(x_matrix[:, startIndex:T], particles[:, startIndex:T], color="black", s=10)

    # plot geneology
    x_star = np.zeros(T)
    for i in range(T - 1, startIndex - 1, -1):
        for j in range(N):
            x_star[i] = particles[j, i]
            # print(range(startIndex,(i)))
            for t in range(startIndex, i):
                x_star[t] = particles[B[j, t], t]
            x_dim = list(range(startIndex, i + 1))
            plt.plot(x_dim, x_star[startIndex:i + 1], color="grey")

    # plot geneology of survived
    x_star = np.zeros(lengthGeneology)
    for j in range(N):
        index = 0
        for t in range(startIndex, T):
            x_star[index] = particles[B[j, t], t]
            index = index + 1

        x_dim = list(range(startIndex, T))
        #plt.plot(x_dim, x_star, color='red')
        if(j==N-1):
            plt.plot(x_dim, x_star, color='red')

    plt.show()


if __name__ == '__main__':
    # r = resampling([.4,.2,.1,.3])
    # print(r)
    T = 100
    x_axis = list(range(1, T + 1))
    x, y = generateData(R=1.0, Q=0.1, T=T)
    plt.plot(x_axis, x)
    plt.show()
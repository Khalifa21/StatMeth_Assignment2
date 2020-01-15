######################################################################################
#
# Author : Niharika Gauraha
#          Uppsala University, Uppsala
#          Email : niharika.gauraha@farmbio.uu.se
#
#
#####################################################################################

# particle Gibbs sampler

# Input:
#   param - state parameters
#   y - measurements
#   x0 - initial state
#   M - number of MCMC runs
#   N - number of particles
#   resamplingMethod - resampling methods:
#     multinomical and systematics resampling methods are supported
# Output:
#       The function returns the sample paths of (q, r, x_{1:T})

# import matplotlib
# matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import sys
import scipy.stats as stats
from csmc import CPF


class PG(CPF):
    def __init__(self, f, g, Q=0, R=0, x0=0, x_ref=None, ancestorSampling=False):
        """ This method is called when you create an instance of the class."""
        # Stop, if input parameters are NULL
        if (f is None) or (g is None):
            sys.exit("Error: the input parameters are NULL")
        self.q = None
        self.r = None
        CPF.__init__(self, f, g, Q, R, x0, x_ref, ancestorSampling)

    def simulate(self, y, QInit, RInit, prior_a, prior_b, N=100, M=100, resamplingMethod="multi"):
        # Stop, if input parameters are NULL
        if y is None:
            sys.exit("Error: the input parameters are NULL")

        # Number of states
        self.T = len(y)
        T = self.T
        # Initialize the state parameters

        q = np.zeros(M)
        r = np.zeros(M)
        X = np.zeros((M, T))

        q[0] = QInit
        r[0] = RInit
        x_ref = np.zeros(T)
        # Initialize the state by running a CPF
        self.Q = q[0]
        self.R = r[0]

        self.generateWeightedParticles(y=y, x_ref=x_ref, N=N, resamplingMethod=resamplingMethod)
        X[0, :] = self.sampleStateTrajectory()

        # Run MCMC loop
        for k in range(1, M):
            # Sample the parameters (inverse gamma posteriors)
            seq_t_1 = np.array(range(0, T - 1))
            err_q = X[k - 1, 1:T] - self.f(X[k - 1, seq_t_1], seq_t_1)
            err_q = sum(err_q ** 2)
            q[k] = stats.invgamma.rvs(a=prior_a + (T - 1) / 2, scale=(prior_b + err_q / 2), size=1)
            err_r = y - self.g(X[k - 1, :])
            err_r = sum(err_r ** 2)
            r[k] = stats.invgamma.rvs(a=prior_a + T / 2, scale=(prior_b + err_r / 2), size=1)
            # Run CPF
            self.Q = q[k]
            self.R = r[k]
            self.generateWeightedParticles(y=y, x_ref=X[k - 1, :], N=N, resamplingMethod=resamplingMethod)
            X[k, :] = self.sampleStateTrajectory()

        self.q = q.copy()
        self.r = r.copy()
        return X[k, :]

    def sampleProcessNoise(self):
        if self.q is None:
            sys.exit("call simulate method first")
        return self.q

    def sampleMeasurementNoise(self):
        if self.r is None:
            sys.exit("call simulate method first")
        return self.r


if __name__ == '__main__':
    # Set up some parameters
    N = 100  # Number of particles
    T = 200  # Length of data record

    def stateTransFunc(x, t=0):
        return .5 * x

    def transferFunc(x):
        return x

    f1 = stateTransFunc
    g1 = transferFunc
    x_ref = np.zeros(T)

    # Generate data
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


    x, y = generateData(R=1, Q=1, T=T)

    RInit = .1
    QInit = .1
    prior_a = .01
    prior_b = .01
    M = 2000
    burnIn = int(M * .3)
    burnIn = 1000
    pg = PG(f1, g1, x0=0, ancestorSampling=False)
    x_mult = pg.simulate(y, QInit, RInit, prior_a, prior_b, N=10, M=M)
    q_trace = pg.sampleProcessNoise()
    r_trace = pg.sampleMeasurementNoise()

    nBins = int(np.floor(np.sqrt(M - burnIn)))
    grid = np.arange(burnIn, M, 1)
    q_trace = q_trace[burnIn:M]

    print("xx")

    # Plot the parameter posterior estimate (solid black line = posterior mean)
    plt.subplot(3, 1, 1)
    plt.hist(q_trace, nBins, normed=1, facecolor='#7570B3')
    plt.xlabel("Q")
    plt.ylabel("posterior density estimate")
    plt.axvline(np.mean(q_trace), color='k')

    # Plot the trace of the Markov chain after burn-in (solid black line = posterior mean)
    plt.subplot(3, 1, 2)
    plt.plot(grid, q_trace, color='#7570B3')
    plt.xlabel("iteration")
    plt.ylabel("Q")
    plt.axhline(np.mean(q_trace), color='k')

    # Plot the autocorrelation function
    plt.subplot(3, 1, 3)
    macf = np.correlate(q_trace - np.mean(q_trace), q_trace - np.mean(q_trace), mode='full')
    idx = int(macf.size / 2)
    macf = macf[idx:]
    macf = macf[0:100]
    macf /= macf[0]
    grid = range(len(macf))
    plt.plot(grid, macf, color='#7570B3')
    plt.xlabel("lag")
    plt.ylabel("ACF of Q")

    plt.show()

    plt.clf()
    nBins = int(np.floor(np.sqrt(M - burnIn)))
    grid = np.arange(burnIn, M, 1)
    r_trace = r_trace[burnIn:M]

    # Plot the parameter posterior estimate (solid black line = posterior mean)
    plt.subplot(3, 1, 1)
    plt.hist(r_trace, nBins, normed=1, facecolor='#1B9E77')
    plt.xlabel("R")
    plt.ylabel("posterior density estimate")
    plt.axvline(np.mean(r_trace), color='k')

    # Plot the trace of the Markov chain after burn-in (solid black line = posterior mean)
    plt.subplot(3, 1, 2)
    plt.plot(grid, r_trace, color='#1B9E77')
    plt.xlabel("iteration")
    plt.ylabel("R")
    plt.axhline(np.mean(r_trace), color='k')

    # Plot the autocorrelation function
    plt.subplot(3, 1, 3)
    macf = np.correlate(r_trace - np.mean(r_trace), r_trace - np.mean(r_trace), mode='full')
    idx = int(macf.size / 2)
    macf = macf[idx:]
    macf = macf[0:100]
    macf /= macf[0]
    grid = range(len(macf))
    plt.plot(grid, macf, color='#1B9E77')
    plt.xlabel("lag")
    plt.ylabel("ACF of R")

    plt.show()

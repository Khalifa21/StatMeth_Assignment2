######################################################################################
#
# Author : Niharika Gauraha
#          Uppsala University, Uppsala
#          Email : niharika.gauraha@farmbio.uu.se
#
#
#####################################################################################

# Sequential Importance Sampling
# Input:
#   f,g,Q,R   - state parameters
#   y       - measurements
#   x0      - initial state
#   x_ref   - reference trajecory
#   N       - number of particles
#   resamplingMethod - resampling methods:
#     multinomial, stratified and systematic resampling methods are supported

# Output:
#   x_star  - sample from target distribution

import matplotlib.pyplot as plt
import numpy as np
import sys
import utils

class SequentialImportanceSampling(object):
    def __init__(self, f, g, Q, R, x0=0, x_ref=None):
        """ """
        self.f = f
        self.g = g  # current global model
        self.Q = Q
        self.R = R
        self.x0 = x0
        self.x_ref = x_ref
        self.resampling = 'multi'
        self.particles = None
        self.normalisedWeights = None
        self.B = None
        self.T = 0
        self.N = 10
        self.logLikelihood = 0.0

    def generateWeightedParticles(self, y, N=100, resamplingMethod="multi"):
    # Stop, if input parameters are NULL
        if y is None:
            sys.exit("Error: the input parameter is NULL")

        # Number of states
        self.T = len(y)
        self.N = N
        self.logLikelihood = 0

        # aliases
        particles = np.zeros((N, T))
        normalisedWeights = np.zeros((N, T))
        B = np.zeros((N, T))


        # set resampling method
        if resamplingMethod == 'systematic':
            self.resampling = utils.systematic_resampling
        elif resamplingMethod == 'stratified':
            self.resampling = utils.stratified_resampling
        else:
            self.resampling = utils.multinomial_resampling

        # Init state, at t=0
        particles[:, 0] = self.x0  # Deterministic initial condition

        # weighting step at t=0
        ypred = self.g(particles[:, 0])
        logweights = -(1 / (2 * self.R)) * (ypred - y[0]) ** 2
        max_weight = max(logweights)  # Subtract the maximum value for numerical stability
        w_p = np.exp(logweights - max_weight)
        normalisedWeights[:, 0] = w_p / sum(w_p)  # Save the normalized weights
        B[:, 0] = list(range(N))

        # accumulate the log-likelihood
        self.logLikelihood = self.logLikelihood + max_weight + np.log(sum(w_p)) - np.log(N)

        for t in range(1, self.T):
            xpred = self.f(particles[:, t - 1], t - 1)

            # propogation step
            particles[:, t] = xpred + np.sqrt(self.Q) * np.random.normal(size=N)

            # weighting step
            ypred = self.g(particles[:, t])
            logweights = -(1 / (2 * self.R)) * (y[t] - ypred) ** 2
            max_weight = max(logweights)  # Subtract the maximum value for numerical stability
            w_p = np.exp(logweights - max_weight)
            w_p = np.multiply(normalisedWeights[:, t-1], w_p)
            normalisedWeights[:, t] = w_p / sum(w_p)  # Save the normalized weights


        B[:, T-1] = list(range(N))
        B = B.astype(int)
        self.particles = particles
        self.normalisedWeights = normalisedWeights
        self.B = B


if __name__=='__main__':
    # Set up some parameters
    N = 100  # Number of particles
    T = 100  # Length of data record


    def stateTransFunc(x, t):
        t = t+1
        return 0.5 * x + 25 * x / (1 + x ** 2) + 8 * np.cos(1.2 * t)

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

    R = 1.
    Q = 0.1
    f = stateTransFunc
    g = transferFunc
    # Generate data
    x, y = utils.generateData(R = R, Q = Q, T=T)

    sis = SequentialImportanceSampling(f, g, Q=Q, R=R, x0=0)
    sis.generateWeightedParticles(y)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.hist(sis.normalisedWeights[:, 1])
    ax2.hist(sis.normalisedWeights[:, 10])
    ax3.hist(sis.normalisedWeights[:, 50])

    #plt.plot(range(N), sis.normalisedWeights[:, 10])

    plt.show()

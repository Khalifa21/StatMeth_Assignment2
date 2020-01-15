######################################################################################
#
# Author : Niharika Gauraha
#          Uppsala University, Uppsala
#          Email : niharika.gauraha@farmbio.uu.se
#
#
#####################################################################################

# Conditional particle filter (CPF) or
# Conditional SMC (CSMC)
# Input:
#   f,g,Q,R   - state parameters
#   y       - measurements
#   x0      - initial state
#   N       - number of particles
#   resamplingMethod - resampling methods:
#     multinomial, stratified and systematic resampling methods are supported

# Output:
#   x_star  - sample from target distribution

import matplotlib.pyplot as plt
import numpy as np
import utils
import sys
from baseParticleFilter import BaseParticleFilter


class CPF(BaseParticleFilter):
    def __init__(self, f, g, Q, R, x0=0, x_ref=None, ancestorSampling=False):
        """ This method is called when you create an instance of the class."""
        # Stop, if input parameters are NULL
        if (f is None)  or (g is None):
            sys.exit("Error: the input parameters are NULL")
        self.AS = ancestorSampling
        BaseParticleFilter.__init__(self, f, g, Q, R, x0, x_ref)

    def generateWeightedParticles(self, y, x_ref, N=100, resamplingMethod="multi"):
    # Stop, if input parameters are NULL
        if y is None:
            sys.exit("Error: the input parameter is NULL")

        # Number of states
        self.T = len(y)
        self.N = N

        # aliases
        T = self.T
        particles = np.zeros((N, T))
        normalisedWeights = np.zeros((N, T))
        B = np.zeros((N, T))
        logLikelihood = 0

        # set resampling method
        if resamplingMethod == 'systematic':
            self.resampling = utils.systematic_resampling
        elif resamplingMethod == 'stratified':
            self.resampling = utils.stratified_resampling
        else:
            self.resampling = utils.multinomial_resampling

        # Init state, at t=0
        particles[:, 0] = self.x0  # Deterministic initial condition
        normalisedWeights[:, 0] = 1/N  # Save the normalized weights
        B[:, 0] = list(range(N))

        for t in range(1, T):
            # resampling step
            newAncestors = self.resampling(normalisedWeights[:, t-1])
            xpred = self.f(particles[:, t - 1], t - 1)
            logweights = -(1 / (2 * self.R)) * (y[t] - self.g(xpred[newAncestors])) ** 2
            max_weight = max(logweights)
            # Subtract the maximum value for numerical stability
            new_weights = np.exp(logweights - max_weight)
            w = new_weights / sum(new_weights)  # Save the normalized weights

            ancestors = self.resampling(w)
            newAncestors = newAncestors[ancestors]
            newAncestors[N - 1] = N - 1

            if self.AS:
                # Ancestor sampling
                m = -(1 / (2 * self.Q)) * (x_ref[t] - xpred) ** 2
                const = max(m)  # Subtract the maximum value for numerical stability
                w_as = np.exp(m - const)
                w_as = w * w_as
                w_as = w_as / sum(w_as)  # Save the normalized weights
                newAncestors[N - 1] = np.where(np.random.uniform(size=1) < np.cumsum(w_as))[0][0]
                #print(newAncestors[N - 1] )

            B[:, t - 1] = newAncestors

            # accumulate the log-likelihood
            logLikelihood = logLikelihood + max_weight + np.log(sum(new_weights)) - np.log(N)

            newAncestors = newAncestors.astype(int)
            # propogation step
            particles[:, t] = xpred[newAncestors] + np.sqrt(self.Q) * np.random.normal(size=N)
            particles[N - 1, t] = x_ref[t]

            # weighting step
            ypred = self.g(particles[:, t])
            logweights = -(1 / (2 * self.R)) * (y[t] - ypred) ** 2
            max_weight = max(logweights)  # Subtract the maximum value for numerical stability
            new_weights = np.exp(logweights - max_weight) / w[ancestors]
            normalisedWeights[:, t] = new_weights / sum(new_weights)

        B[:, T - 1] = list(range(N))

        self.particles = particles[:]
        self.normalisedWeights = normalisedWeights[:]
        self.B = B.astype(int)
        self.logLikelihood = logLikelihood

    # Given theta estimate states using PG
    def iteratedCPF(self, y, x_ref, N = 100, resamplingMethod = "multi", iter = 100):

        for i in range(iter):
            self.generateWeightedParticles(y, x_ref, N, resamplingMethod)
            x_ref = self.sampleStateTrajectory()

        return x_ref



if __name__=='__main__':
    # Set up some parameters
    N = 10  # Number of particles
    T = 21  # Length of data record
    f1 = utils.stateTransFunc
    g1 = utils.transferFunc
    # R = 1.
    # Q = 0.1
    x_ref = np.zeros(T)
    # Generate data
    x, y = utils.generateData(R=1.0, Q=0.1, T=T)

    def stateTransFunc(x, t=0):
        return .1 + np.sin(x)


    def transferFunc(x):
        return x

    f1 = stateTransFunc
    g1 = transferFunc
    # R = 1.
    # Q = 0.1
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

    x, y = generateData(R=1, Q=.1, T=T)

    pf = CPF(f1, g1, Q=0.1, R=1, x0=0)
    pf.generateWeightedParticles(y, x, N=N)
    particles = np.zeros((N, T))
    for t in range(T):
        particles[:,t] = list(range(1,N+1))

    #utils.particleGeneologyAll(particles, pf.B, T-1)
    #utils.particleGeneologyAll(pf.particles, pf.B, T-1)

    '''
    x_mult = pf.sampleStateTrajectory()
    pf.generateWeightedParticles(y, x_ref, resamplingMethod='systematic')
    x_sys = pf.sampleStateTrajectory()
    pf.generateWeightedParticles(y, x_ref, resamplingMethod='stratified')
    x_stra = pf.sampleStateTrajectory()
    plt.plot(x, marker='o', label='Real States', markersize=3)
    plt.plot(x_mult, label='CPF+Multinomial Filtered States')
    plt.plot(x_sys, label='CPF+Systematic Filtered States')
    plt.plot(x_stra, label='CPF+Stratified Filtered States')
    plt.legend()
    plt.show()
    x_pg = pf.iteratedCPF(y, x_ref, N=10, resamplingMethod='systematic')
    plt.plot(x, marker='o', label='Real States', markersize=3)
    plt.plot(x_pg, label='PG+Systematic Filtered States')
    #plt.legend()
    #plt.show()
    pf = CPF(f1, g1, Q=0.1, R=1, x0=0, ancestorSampling=True)
    # with ancestor sampling
    x_pg = pf.iteratedCPF(y, x_ref, N=10, resamplingMethod='systematic')
    #plt.plot(x, marker='o', label='Real States', markersize=3)
    plt.plot(x_pg, label='PGAS+Systematic Filtered States')
    plt.legend()
    plt.show()
    '''

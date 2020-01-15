import utils
import numpy as np

class BaseParticleFilter(object):
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

    def generateWeightedParticles(self):  # and register
        print("generate weighted particles")

    def plotGeneology(self, type, lengthGeneology):  # and register
        print("plot Geneoloty")
        if type == 'all':
            utils.particleGeneologyAll(self.particles, self.B, lengthGeneology = lengthGeneology)
        else:
            utils.particleGeneology(self.particles, self.B, lengthGeneology = lengthGeneology)

    def sampleStateTrajectory(self):
        if self.particles is None:
            print("call generateWeightedParticles first")
            exit(0)

        x_star = np.zeros(self.T)
        J = np.where(np.random.uniform(size=1) < np.cumsum(self.normalisedWeights[:, self.T - 1]))[0][0]

        for t in range(self.T):
            x_star[t] = self.particles[self.B[J, t], t]
        return x_star

    def getLoglikelihood(self):
        return self.logLikelihood

    def setParameters(self, f, g, Q, R):
        self.f = f
        self.g = g  # current global model
        self.Q = Q
        self.R = R

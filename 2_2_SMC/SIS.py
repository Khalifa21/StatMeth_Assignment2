from math import *
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import normal
import DataGenerator

T = 100
phi = 1.
sigma = 0.16
beta = 0.64
N = 100

def SIS(y,T,N,x0, phi, sigma, beta):
    """ N : nb particles, T : nb of samples"""
    particles = np.zeros((N, T))
    normalisedWeights = np.zeros((N, T))

    # Init state
    particles[:, 0] = x0  # Deterministic initial condition
    w = np.zeros(N)
    for i in range(N):
        w[i] = normal(0,beta*exp(particles[i,0]))
    
    normalisedWeights[:, 0] = w / sum(w)

    for t in range(1,T):
        for i in range(N):
            particles[i, t] = normal(phi*particles[i, t - 1],sigma**2)

        w = np.zeros(N)
        for i in range(N):
            w[i] = normal(0,beta*exp(particles[i,t]))
    
        w = np.multiply(normalisedWeights[:, t-1], w)
        normalisedWeights[:, t] = w / sum(w)

    #point estimate
    point_x = sum(np.multiply(normalisedWeights[-1], particles[-1]))/N

    #variance normalized weights
    var_weights = np.var(normalisedWeights[-1])

    return(normalisedWeights,particles, point_x, var_weights)

def log_SIS(y,T,N,x0, phi, sigma, beta):
    """ N : nb particles, T : nb of samples"""
    particles = np.zeros((N, T))
    normalisedWeights = np.zeros((N, T))

    # Init state
    particles[:, 0] = x0  # Deterministic initial condition
    w = np.zeros(N)
    for i in range(N):
        w[i] = normal(0,beta*exp(particles[i,0]))
    
    logweights = -(1 /2) * (y[0] - w) ** 2
    max_weight = max(logweights)  # Subtract the maximum value for numerical stability
    w_p = np.exp(logweights - max_weight)
    normalisedWeights[:, 0] = w_p / sum(w_p)

    for t in range(1,T):
        for i in range(N):
            particles[i, t] = normal(phi*particles[i, t - 1],sigma**2)

        w = np.zeros(N)
        for i in range(N):
            w[i] = normal(0,beta*exp(particles[i,t]))
        
        logweights = -(1 /2) * (y[t] - w) ** 2
        max_weight = max(logweights)  # Subtract the maximum value for numerical stability
        w_p = np.exp(logweights - max_weight)
        w_p = np.multiply(normalisedWeights[:, t-1], w_p)
        normalisedWeights[:, t] = w_p / sum(w_p)

    #point estimate
    point_x = sum(np.multiply(normalisedWeights[-1], particles[-1]))/N

    #variance normalized weights
    var_weights = np.var(normalisedWeights[-1])

    return(normalisedWeights,particles, point_x, var_weights)

def main():
    x0 = normal(0,sigma**2)
    x,y = DataGenerator.SVGenerator2(phi,sigma, beta,T)

    def point_plot():
        X_point = []
        for n in range(10,50,1):
            norm_w,particles,point_x, var_weights=SIS(y,T,n,x[0], phi, sigma, beta)
            X_point.append(1/2*(point_x-x[-1])**2)

        plt.plot(X_point)
        plt.show()

    def var_plot():
        X_point = []
        for n in range(10,50,1):
            norm_w,particles,point_x, var_weights=log_SIS(y,T,n,x[0], phi, sigma, beta)
            X_point.append(var_weights)

        plt.plot(X_point)
        plt.show()

    def weights_plot():
        norm_w,particles,point_x,var_weights=log_SIS(y,T,N,x[0], phi, sigma, beta)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.hist(norm_w[:, 1])
        ax2.hist(norm_w[:, 10])
        ax3.hist(norm_w[:, 50])
        ax1.set_title('normalized weights iter 1')
        ax2.set_title('normalized weights iter 10')
        ax3.set_title('normalized weights iter 50')
        plt.show()

    weights_plot()

if __name__ == "__main__": main()

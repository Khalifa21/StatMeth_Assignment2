from math import *
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import normal
import DataGenerator
import scipy.stats as stats

T = 100
phi = 1.
sigma = 0.16
beta = 0.64
N = 100
sampling_method = "stratified"

def multinomial_resampling(ws):
    N = len(ws)
    
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

        i += 1

    return out

def stratified_resampling(ws):
    N = len(ws)

    # Output array
    out = np.zeros((N,), dtype=int)

    # Find the right ranges
    total = ws[0]
    j = 0
    for i in range(N):
        u = (stats.uniform.rvs() + i - 1) / N
        while j < (N-1) and total < u: ##### CONDITION RAJOUTEE ICI !!
            j += 1
            total += ws[j]

        # Once the right index is found, save it
        out[i] = j

    return out

def BPF(y,T,N,x0, phi, sigma, beta, sampling_method):
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
        # Resampling
        if sampling_method == "multi":
            indexes = multinomial_resampling(normalisedWeights[-1])
        elif sampling_method == "stratified":
            indexes = stratified_resampling(normalisedWeights[-1])
        else:
            sys.exit('Specify a sampling method. Either "muli" or "stratified"')
        resample_particules = particles[-1][indexes]

        for i in range(N):
            particles[i, t] = normal(phi*resample_particules[i],sigma**2)

        w = np.zeros(N)
        for i in range(N):
            w[i] = normal(0,beta*exp(particles[i,t]))
    
        normalisedWeights[:, t] = w / sum(w)

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
            norm_w,particles,point_x, var_weights=BPF(y,T,n,x[0], phi, sigma, beta, sampling_method)
            X_point.append(1/2*(point_x-x[-1])**2)

        plt.plot(X_point)
        plt.show()

    def var_plot():
        X_point = []
        for n in range(10,50,1):
            norm_w,particles,point_x, var_weights=BPF(y,T,n,x[0], phi, sigma, beta, sampling_method)
            X_point.append(var_weights)

        plt.plot(X_point)
        plt.show()

    def weights_plot():
        norm_w,particles,point_x,var_weights=BPF(y,T,N,x[0], phi, sigma, beta, sampling_method)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.hist(norm_w[:, 1])
        ax2.hist(norm_w[:, 10])
        ax3.hist(norm_w[:, 50])
        ax1.set_title('normalized weights iter 1')
        ax2.set_title('normalized weights iter 10')
        ax3.set_title('normalized weights iter 50')
        plt.show()

    weights_plot()
    #point_plot()

if __name__ == "__main__": main()


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
sampling_method = "multi"

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
        while j < (N-1) and total < u: ##### CONDITION j < (N-1) ADDED HERE
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
        w[i] = normal(0,beta*exp(particles[i,0]/2))
    
    normalisedWeights[:, 0] = w / sum(w)

    for t in range(1,T):
        # Resampling
        if sampling_method == "multi":
            indexes = multinomial_resampling(normalisedWeights[:,-1])
        elif sampling_method == "stratified":
            indexes = stratified_resampling(normalisedWeights[:,-1])
        else:
            sys.exit('Specify a sampling method. Either "muli" or "stratified"')
        resample_particules = particles[:,t-1][indexes]

        w = np.zeros(N)
        for i in range(N):
            particles[i, t] = normal(phi*resample_particules[i],sigma)
            w[i] = normal(0,beta*exp(particles[i,t]/2))
    
        normalisedWeights[:, t] = w / sum(w)

    #point estimate
    point_x = sum(np.multiply(normalisedWeights[:,-1], particles[:,-1]))

    #variance normalized weights
    var_weights = np.var(normalisedWeights[:,-1])

    return(normalisedWeights,particles, point_x, var_weights)

def log_BPF(y,T,N,x0, phi, sigma, beta, sampling_method):
    """ N : nb particles, T : nb of samples"""
    particles = np.zeros((N, T))
    normalisedWeights = np.zeros((N, T))

    # Init state
    particles[:, 0] = x0  # Deterministic initial condition
    logweights = np.zeros(N)

    for i in range(N):
        sigma_p = beta*exp(particles[i,0]/2)
        logweights[i] = -np.log(sigma_p*np.sqrt(2*np.pi))-1/2*(particles[i,0]/sigma_p)**2
    
    max_weight = max(logweights)  # Subtract the maximum value for numerical stability
    w_p = np.exp(logweights - max_weight)
    normalisedWeights[:, 0] = w_p / sum(w_p)

    for t in range(1,T):
        # Resampling
        if sampling_method == "multi":
            indexes = multinomial_resampling(normalisedWeights[:,-1])
        elif sampling_method == "stratified":
            indexes = stratified_resampling(normalisedWeights[:,-1])
        else:
            sys.exit('Specify a sampling method. Either "muli" or "stratified"')
        resample_particules = particles[:,t-1][indexes]

        for i in range(N):
            particles[i, t] = normal(phi*resample_particules[i],sigma)

        logweights = np.zeros(N)
        for i in range(N):
            sigma_p = beta*exp(particles[i,t]/2)
            logweights[i] = -np.log(sigma_p*np.sqrt(2*np.pi))-1/2*(particles[i,t]/sigma_p)**2
        
        max_weight = max(logweights)  # Subtract the maximum value for numerical stability
        w_p = np.exp(logweights - max_weight)
        normalisedWeights[:, t] = w_p / sum(w_p)

    #point estimate
    point_x = sum(np.multiply(normalisedWeights[:,-1], particles[:,-1]))

    #variance normalized weights
    var_weights = np.var(normalisedWeights[:,-1])

    return(normalisedWeights, particles, point_x, var_weights)


def main():
    x0 = normal(0,sigma)
    x,y = DataGenerator.SVGenerator2(phi, sigma, beta, T)
    print(x[-1])

    def point_plot():
        for j in range(3):
            X_point = []
            for n in range(10,100,1):
                norm_w,particles,point_x, var_weights=log_BPF(y,T,n,x0, phi, sigma, beta, sampling_method)
                X_point.append(1/2*(point_x-x[-1])**2)
            plt.plot(X_point)
        plt.xlabel('number of samples N')
        plt.ylabel('mse')
        plt.show()

    def var_plot():
        X_point = []
        for n in range(10,200,1):
            norm_w, particles, point_x, var_weights = log_BPF(y,T,n,x0, phi, sigma, beta, sampling_method)
            X_point.append(var_weights)

        plt.plot(X_point)
        plt.show()


    def point_plot2():
        for j in range(3):
            norm_w,particles,point_x,var_weights=log_BPF(y,T,N,x0, phi, sigma, beta, sampling_method)
            X = []
            for i in range(T):
                x_1 = sum(np.multiply(norm_w[:,i], particles[:,i]))
                X.append(x_1)
            plt.plot(X, label = "estimated x "+str(j))
        plt.plot(x, label = "true x")
        plt.legend()
        plt.xlabel("T")
        plt.ylabel("xt")
        plt.show()

    def weights_plot():
        norm_w,particles,point_x,var_weights=log_BPF(y,T,N,x0, phi, sigma, beta, sampling_method)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.hist(norm_w[:, 1])
        ax2.hist(norm_w[:, 10])
        ax3.hist(norm_w[:, 50])
        ax1.set_title('normalized weights iter 1')
        ax2.set_title('normalized weights iter 10')
        ax3.set_title('normalized weights iter 50')
        plt.show()

    #weights_plot()
    point_plot2()
    #var_plot()

if __name__ == "__main__": main()


from math import *
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import normal
import DataGenerator
import scipy.stats as stats, norm

T = 100
phi = 1.
#sigma = 0.16
#beta = 0.64
N = 200
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

def log_BPF(y, T, N, x0, phi, sigma, beta, sampling_method):
    """ N : nb particles, T : nb of samples"""
    particles = np.zeros((N, T))
    normalisedWeights = np.zeros((N, T))
    likelihood = 0

    # Init state
    particles[:, 0] = x0  # Deterministic initial condition
    logweights = np.zeros(N)

    for i in range(N):
        sigma_p = beta*exp(particles[i,0]/2)
        #logweights[i] = -np.log(sigma_p*np.sqrt(2*np.pi))-1/2*(particles[i,0]/sigma_p)**2
        logweights[i] = -1/2*(particles[i,0])**2
    
    max_weight = max(logweights)  # Subtract the maximum value for numerical stability
    w_p = np.exp(logweights - max_weight)
    normalisedWeights[:, 0] = w_p / sum(w_p)

    likelihood += max_weight + np.sum(logweights) - np.log(N)

    for t in range(1,T):
        # Resampling
        if sampling_method == "multi":
            indexes = multinomial_resampling(normalisedWeights[:,-1])
        elif sampling_method == "stratified":
            indexes = stratified_resampling(normalisedWeights[:,-1])
        else:
            sys.exit('Specify a sampling method. Either "muli" or "stratified"')
        resample_particules = particles[:,t-1][indexes]
        print(resample_particules)

        for i in range(N):
            particles[i, t] = normal(phi*resample_particules[i],sigma)

        logweights = np.zeros(N)
        for i in range(N):
            sigma_p = beta*exp(particles[i,t]/2)
            #logweights[i] = -np.log(sigma_p*np.sqrt(2*np.pi))-1/2*(particles[i,t]/sigma_p)**2
            logweights[i] = -1/2*(particles[i,t])**2
        
        max_weight = max(logweights)  # Subtract the maximum value for numerical stability
        w_p = np.exp(logweights - max_weight)
        normalisedWeights[:, t] = w_p / sum(w_p)

    likelihood += max_weight + np.sum(logweights) - np.log(N)

    return(normalisedWeights, particles, likelihood)
    
def log_SIS(y,T,N,x0, phi, sigma, beta):
    """ N : nb particles, T : nb of samples"""
    particles = np.zeros((N, T))
    normalisedWeights = np.zeros((N, T))
    likelihood = 0

    # Init state
    particles[:, 0] = x0  # Deterministic initial condition
    logweights = np.zeros(N)
    for i in range(N):
        sigma_p = beta*exp(particles[i,0]/2)
        logweights[i] = -1/2*(particles[i,0])**2
    
    max_weight = max(logweights)  # Subtract the maximum value for numerical stability
    w_p = np.exp(logweights - max_weight)
    normalisedWeights[:, 0] = w_p / sum(w_p)

    likelihood += max_weight + np.sum(logweights) - np.log(N)

    for t in range(1,T):
        w = np.zeros(N)
        logweights = np.zeros(N)
        for i in range(N):
            sigma_p = beta*exp(particles[i,t]/2)
            particles[i, t] = normal(phi*particles[i, t - 1],sigma)
            #w[i] = normal(0,beta*exp(particles[i,t]/2))
            logweights[i] = -1/2*(particles[i,t])**2
        max_weight = max(logweights)  # Subtract the maximum value for numerical stability
        w_p = np.exp(logweights - max_weight)
        w_p = np.multiply(normalisedWeights[:, t-1], w_p)
        normalisedWeights[:, t] = w_p / sum(w_p)

    #point estimate
    point_x = sum(np.multiply(normalisedWeights[:,-1], particles[:,-1]))/N

    #variance normalized weights
    var_weights = np.var(normalisedWeights[:,-1])

    likelihood += max_weight + np.sum(logweights) - np.log(N)

    return(normalisedWeights,particles, likelihood) 


def main():
    y = []
    with open('y.txt', 'r') as f:
        for line in f:
            y.append(np.float(line.replace('\n', '')))

    def point_plot():
        for j in range(3):
            X_point = []
            for n in range(10,100,1):
                norm_w, particles, likelihood = log_BPF(y,T,n,x0, phi, sigma, beta, sampling_method)
                X_point.append(1/2*(point_x-x[-1])**2)
            plt.plot(X_point)
        plt.xlabel('number of samples N')
        plt.ylabel('mse')
        plt.show()

    def point_plot2():
        for j in range(3):
            norm_w, particles, likelihood=log_BPF(y,T,N,x0, phi, sigma, beta, sampling_method)
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
        norm_w, particles, likelihood = log_BPF(y,T,N,x0, phi, sigma, beta, sampling_method)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.hist(norm_w[:, 1])
        ax2.hist(norm_w[:, 10])
        ax3.hist(norm_w[:, 50])
        ax1.set_title('normalized weights iter 1')
        ax2.set_title('normalized weights iter 10')
        ax3.set_title('normalized weights iter 50')
        plt.show()

    def likelihood_plot():
        #I = [0, 0.2, 0.5, 0.8, 1.2, 1.5, 1.8, 2]
        #for beta in I:
        #    for sigma in I:
        #        x0 = normal(0,sigma)
        #        norm_w, particles, likelihood = log_BPF(y, T, 100, x0, phi, sigma, beta, sampling_method)
        #        print("beta :", beta, "| sigma: ", sigma, " ----> likelihood", likelihood)

        I = [x/100 for x in range(0,200,2)]
        sigma = 0.16
        list_beta = []
        for beta in I:
            x0 = normal(0,sigma)
            norm_w, particles, likelihood = log_BPF(y, T, 200, x0, phi, sigma, beta, sampling_method)
            #norm_w, particles, likelihood = log_SIS(y, T, 100, x0, phi, sigma, beta)
            list_beta.append(likelihood)
        plt.plot(I, list_beta)
        plt.show()

    likelihood_plot()

if __name__ == "__main__": main()


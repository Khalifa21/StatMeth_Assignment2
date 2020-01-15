from math import *
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import normal
import DataGenerator
import scipy.stats as stats


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

def stratified_resampling(ws):
    N = len(ws)

    # Output array
    out = np.zeros((N,), dtype=int)

    # Find the right ranges
    total = ws[0]
    j = 0
    for i in range(N):
        u = (stats.uniform.rvs() + i - 1) / N
        while total < u: 
            j += 1
            total += ws[j]

        # Once the right index is found, save it
        out[i] = j

    return out

def log_SIS(beta, y, N=100):
    sigma = 0.16
    phi = 1

    T = len(y)
    # Number of states
    logLikelihood = 0
    particles = np.zeros((N, T))
    normalisedWeights = np.zeros((N, T))

    # Init state, at t=0
    x0 = normal(0, sigma, N) 
    particles[:, 0] = x0

    # weighting step at t=0
    logweights = np.zeros(N)
    for i in range(N):
        logweights[i] = np.log(stats.norm.pdf(y[0], 0, beta*np.exp(x0[i]/2)))

    max_weight = np.max(logweights)  # Subtract the maximum value for numerical stability
    w_p = np.exp(logweights - max_weight)
    normalisedWeights[:, 0] = w_p / np.sum(w_p)  # Save the normalized weights

    # accumulate the log-likelihood
    logLikelihood = logLikelihood + max_weight + np.log(np.sum(w_p)) - np.log(N)

    for t in range(1, T):
        particles[:, t] = normal(phi*particles[:, t - 1], sigma)
        #xpred = stats.norm.pdf(particles[:,t-1], phi*particles[:, t - 1], sigma)

        # weighting step
        logweights = np.zeros(N)
        for i in range(N):
            if stats.norm.pdf(y[t], 0, beta*np.exp(particles[i, t]/2)) == 0 :
                break
            else:
                logweights[i] = np.log(stats.norm.pdf(y[t], 0, beta*np.exp(particles[i, t]/2)))

        max_weight = np.max(logweights)  # Subtract the maximum value for numerical stability
        w_p = np.exp(logweights - max_weight)
        w_p = np.multiply(normalisedWeights[:, t-1], w_p)
        normalisedWeights[:, t] = w_p / np.sum(w_p)  # Save the normalized weights
     
        logLikelihood = logLikelihood + max_weight + np.log(np.sum(w_p)) - np.log(N)

    return(normalisedWeights, particles, logLikelihood) 

def log_BPF(beta, sigma, y, N, sampling_method="multi"):

    phi = 1

    T = len(y)

    logLikelihood = 0
    particles = np.zeros((N, T))
    normalisedWeights = np.zeros((N, T))

    # Init state, at t=0
    x0 = normal(0, sigma, N) 
    particles[:, 0] = x0

    logweights_0 = np.zeros(N)
    #for i in range(N):
    #    logweights_0[i] = np.log(stats.norm.pdf(y[0], 0, beta*np.exp(x0[i]/2)))
    logweights_0 = stats.norm.logpdf(y[0], 0, beta*np.exp(x0/2))

    max_weight_0 = np.max(logweights_0)  # Subtract the maximum value for numerical stability
    w_p_0 = np.exp(logweights_0 - max_weight_0)
    normalisedWeights[:, 0] = w_p_0 / np.sum(w_p_0)  # Save the normalized weights

    # accumulate the log-likelihood
    logLikelihood = logLikelihood + max_weight_0 + np.log(sum(w_p_0)) - np.log(N)

    for t in range(1, T):
        # Resampling
        if sampling_method == "multi":
            indexes = multinomial_resampling(normalisedWeights[:,t-1])
        elif sampling_method == "stratified":
            indexes = stratified_resampling(normalisedWeights[:,t-1])
        else:
            sys.exit('Specify a sampling method. Either "muli" or "stratified"')
        resample_particules = particles[:,t-1][indexes]

        for i in range(N):
            particles[i, t] = normal(phi*resample_particules[i],sigma)

        # weighting step
        logweights = np.zeros(N)
        for i in range(N):
            #if stats.norm.pdf(y[t], 0, beta*np.exp(particles[i, t]))==0:
            #    break
            #else:
            logweights[i] = stats.norm.logpdf(y[t], 0, beta*np.exp(particles[i, t]))
       
        max_weight = np.max(logweights)  # Subtract the maximum value for numerical stability
        w_p = np.exp(logweights - max_weight)
        #w_p = np.multiply(normalisedWeights[:, t-1], w_p)
        normalisedWeights[:, t] = w_p / np.sum(w_p)  # Save the normalized weights
     
        logLikelihood = logLikelihood + max_weight + np.log(np.sum(w_p)) - np.log(N)

    return(normalisedWeights,particles, logLikelihood) 

def main():
    y = []

    with open('y.txt', 'r') as f:
        for line in f:
            y.append(np.float(line.replace('\n', '')))

    x = []
    with open('x.txt', 'r') as f:
        for line in f:
            x.append(np.float(line.replace('\n', '')))

    T = len(y)-1

    def point_plot(algo):
        beta = 0.64
        for j in range(1):
            X_point = []
            step = 2
            for n in range(10,100,step):
                norm_w, particles, likelihood = algo(beta, y, n)
                point_x = sum(np.multiply(norm_w[:,-1], particles[:,-1]))
                X_point.append(1/2*(point_x-x[-1])**2)
            plt.plot(X_point)
        plt.xlabel('number of samples N/'+str(step))
        plt.ylabel('mse')
        plt.show()

    def point_plot2(algo):
        T = 100
        beta = 0.64
        for j in range(3):
            norm_w, particles, likelihood=algo(beta, y, 100)
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

    def point_plot3():
        beta = 0.64
        
        norm_w, particles, likelihood=log_SIS(beta, y, 100)
        X = []
        for i in range(T):
            x_1 = sum(np.multiply(norm_w[:,i], particles[:,i]))
            X.append(x_1)
        print(len(X))
        plt.plot(X, label = "log_SIS")

        norm_w, particles, likelihood=log_BPF(beta, y, 100, 'multi')
        X = []
        for i in range(T):
            x_1 = sum(np.multiply(norm_w[:,i], particles[:,i]))
            X.append(x_1)
        print(len(X))
        plt.plot(X, label = "log_BPF multi")

        norm_w, particles, likelihood=log_BPF(beta, y, 100, 'stratified')
        X = []
        for i in range(T):
            x_1 = sum(np.multiply(norm_w[:,i], particles[:,i]))
            X.append(x_1)
        print(len(X))
        plt.plot(X, label = "log_BPF stratified")


        plt.plot(x, label = "true x")
        plt.legend()
        plt.xlabel("T")
        plt.ylabel("xt")
        plt.show()

    def weights_plot(algo):
        beta = 0.64
        norm_w, particles, likelihood = algo(beta, y, 100)
        print('var weights = ', np.var(norm_w[T]))
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.hist(norm_w[:, 1])
        ax2.hist(norm_w[:, 10])
        ax3.hist(norm_w[:, 50])
        ax1.set_title('normalized weights iter 1')
        ax2.set_title('normalized weights iter 10')
        ax3.set_title('normalized weights iter 50')
        plt.show()

    def var_plot():
        X_point = []
        beta = 0.64
        for n in range(10,50,1):
            norm_w, particles, likelihood = algo(beta, y, 100)
            var_weights = np.var(norm_w[T])
            X_point.append(var_weights)
        plt.plot(X_point)
        plt.show()

    def likelihood_plot(algo):
        I = [x/10 for x in range(2,20,1)]
        list_beta = []
        for beta in I:
            print("beta = ", beta)
            norm_w, particles, likelihood = algo(beta, 0.16, y, 100)
            #norm_w, particles, likelihood = algo(beta, 0.16, y, 100)
            list_beta.append(likelihood)
        plt.plot(I, list_beta)
        plt.show()

    def likelihood_plot2(algo):
        I_beta = [0.2, 0.4, 0.6, 0.8, 1, 1.3, 1.5, 2]
        I_sigma = [0.1, 0.2, 0.5, 0.8,  1, 1.3, 1.5, 2]
        list_bs = np.zeros((len(I_beta),len(I_sigma)))
        for counter_beta, beta in enumerate(I_beta):
            print("beta ", beta)
            for counter_sigma, sigma in enumerate(I_sigma):
                print("sigma ", sigma)
                norm_w, particles, likelihood = algo(beta, sigma, y, 100)
                list_bs[counter_beta, counter_sigma] = likelihood
                print("likelihood : ", likelihood)


        fig, ax = plt.subplots()
        im = ax.imshow(list_bs)

        # We want to show all ticks...
        ax.set_xticks(np.arange(len(I_sigma)))
        ax.set_yticks(np.arange(len(I_beta)))
        # ... and label them with the respective list entries
        ax.set_xticklabels(I_sigma)
        ax.set_yticklabels(I_beta)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        ax.set_title("Log Likelihood variations with beta and sigma")
        fig.tight_layout()
        plt.show()

    def variance_likelihood(algo, param = "T"):
        sigma = 0.16
        beta = 0.64
        phi = 1.
        step = 2

        if param == "N":
            T = 100
            x,y = DataGenerator.SVGenerator(phi, sigma, beta, T)
            list_likelihood = []
            for n in range(10,500,step):
                norm_w, particles, likelihood = algo(beta, sigma, y, n)
                list_likelihood.append(likelihood)
                print(n)
            plt.plot(list_likelihood)
            plt.xlabel('N/'+str(step))
        
        if param == "T":
            N = 100
            list_likelihood = []
            for t in range(10,200,step):
                print(t)
                x,y = DataGenerator.SVGenerator(phi, sigma, beta, t)
                norm_w, particles, likelihood = algo(beta, sigma, y, N)
                list_likelihood.append(likelihood)
            plt.plot(list_likelihood)
            plt.xlabel('T/'+str(step))
        
        plt.ylabel('log likelihood')
        plt.show()

    algo = log_BPF
    #algo(0.64, y)
    #likelihood_plot(algo)
    #point_plot2(algo)
    #weights_plot(algo)
    #point_plot(algo)
    #point_plot3()
    #likelihood_plot2(algo)
    variance_likelihood(algo)

if __name__ == "__main__": main()
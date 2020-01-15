from math import *
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import normal
import DataGenerator
import scipy.stats as stats
import sys
import math
from scipy.stats import invgamma
from tqdm import tqdm
T = 100

def SIS(beta, y, N=100):
    T = 100
    phi = 1
    sigma = 0.16
    l = 0

    particles = np.zeros((N, T))
    norm_w = np.zeros((N,T))

    x0 = normal(0, sigma, N)
    particles[:,0] = x0

    alpha = np.zeros(N)
    for i in range(N):
        alpha[i] = stats.norm.pdf(y[0], 0, beta*exp(x0[i]/2))

    w = alpha
    w = w / np.sum(w)
    
    norm_w[:,0] = w
    l += np.log(sum(w))

    for t in range(1, T):
        alpha = np.zeros(N)
        for i in range (N):
            particles[i, t] = normal(phi*particles[i, t - 1],sigma)
            alpha[i] = stats.norm.pdf(y[t], 0, beta*exp(particles[i,t]/2))
        w = np.multiply(w,alpha)
        w = w/np.sum(w)
        norm_w[:,t] = w
        l += np.log(sum(w))

    print(l)

    return(norm_w, particles, l)

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
            if stats.norm.pdf(y[t], 0, beta*np.exp(particles[i, t]/2)) == 0:
                break
            else:
                logweights[i] = np.log(stats.norm.pdf(y[t], 0, beta*np.exp(particles[i, t]/2)))

        max_weight = np.max(logweights)  # Subtract the maximum value for numerical stability
        w_p = np.exp(logweights - max_weight)
        w_p = np.multiply(normalisedWeights[:, t-1], w_p)
        normalisedWeights[:, t] = w_p / np.sum(w_p)  # Save the normalized weights
     
        logLikelihood = logLikelihood + max_weight + np.log(np.sum(w_p)) - np.log(N)

    return(normalisedWeights, particles, logLikelihood) 


def log_BPF(phi, beta, sigma ,y, N, sampling_method = "multi"):
    # sigma = 0.16
    #phi = 1

    logLikelihood = 0
    particles = np.zeros((N, T))
    normalisedWeights = np.zeros((N, T))

    # Init state, at t=0
    x0 = normal(0, sigma, N) 
    particles[:, 0] = x0

    logweights_0 = np.zeros(N)
    for i in range(N):
        logweights_0[i] = np.log(stats.norm.pdf(y[0], 0, beta*np.exp(x0[i]/2)))

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
            #sigma_p = beta*np.exp(particles[i, t])
            #logweights[i] = -1/2*(y[t]/sigma_p)**2
            if stats.norm.pdf(y[t], 0, beta*np.exp(particles[i, t]))==0:
                break
            else:
                logweights[i] = np.log(stats.norm.pdf(y[t], 0, beta*np.exp(particles[i, t])))
        max_weight = np.max(logweights)  # Subtract the maximum value for numerical stability
        w_p = np.exp(logweights - max_weight)
        #w_p = np.multiply(normalisedWeights[:, t-1], w_p)
        normalisedWeights[:, t] = w_p / np.sum(w_p)  # Save the normalized weights
     
        logLikelihood = logLikelihood + max_weight + np.log(np.sum(w_p)) - np.log(N)

    return(normalisedWeights,particles, logLikelihood) 


# def inv_gamma(a,b,x):
#     return ((b^a)/math.lgamma(a))* x^(-a-1)*exp(-b/x)

def PMH(y, x0, phi=1, N=100, iterations=100):
    # A = np.zeros(iterations)
    beta = np.zeros(iterations)
    sigma = np.zeros(iterations)
    z = np.zeros(iterations)
    step1 = 0.01

    # A[0] = .5
    beta[0] = .5
    sigma[0] = .1


    z[0] = log_BPF(phi,sqrt(beta[0]), sqrt(sigma[0]) ,y, N, sampling_method = "multi")[2]

    for i in tqdm(range(1,iterations)):
        # Propose new parameters
        # A_proposed = A[i - 1] + step1 * np.random.normal(size=1)
        beta_proposed = beta[i-1] + step1 * np.random.normal(size=1)
        sigma_proposed = sigma[i-1] + step1 * np.random.normal(size=1)
        # while (abs(A_proposed) > 1):
        #     A_proposed = A[i - 1] + step1 * np.random.normal(size=1)
        while (beta_proposed > 1 or beta_proposed <= 0):
            beta_proposed = beta[i - 1] + step1 * np.random.normal(size=1)  # inv_gamma(.01,.01,x0)
        while (sigma_proposed > 1 or sigma_proposed <= 0):
            sigma_proposed =sigma[i - 1] + step1 * np.random.normal(size=1)  # inv_gamma(.01,.01,x0)

        # Run a PF to evaluate the likelihood
        # z_hat = log_BPF(y=y, A=A_proposed, Q=Q, R=R, x0=x0, N=N)
        z_hat = log_BPF(phi, sqrt(beta_proposed), sqrt(sigma_proposed), y, N, sampling_method="multi")[2]
        # Sample from uniform
        u = stats.uniform.rvs()

        # Compute the acceptance probability
        # numerator = z_hat + stats.norm.logpdf(A_proposed)
        numerator = z_hat + log(invgamma.pdf(a=.01,scale=.01, x=beta_proposed)) + log(invgamma.pdf(a=.01, scale=.01, x=sigma_proposed))
        # denominator = z[i - 1] + stats.norm.logpdf(A[i - 1])
        denominator = z[i - 1] + log(invgamma.pdf(a=.01,scale=.01, x=beta[i-1])) + log(invgamma.pdf(a=.01, scale=.01, x=sigma[i-1]))

        # Acceptance probability
        alpha = np.exp(numerator - denominator)
        # alpha = min(1,alpha)
        # Set next state with acceptance probability
        if u <= alpha:
            z[i] = z_hat
            # A[i] = A_proposed
            beta[i] = beta_proposed
            sigma[i] = sigma_proposed

        else:
            z[i] = z[i-1]
            # A[i] = A[i-1]
            beta[i] = beta[i-1]
            sigma[i] = sigma[i-1]

    # return A
    return beta, sigma, z

def main():
    y = []

    with open('y.txt', 'r') as f:
        for line in f:
            y.append(np.float(line.replace('\n', '')))

    x = []
    with open('x.txt', 'r') as f:
        for line in f:
            x.append(np.float(line.replace('\n', '')))

    T = len(y) - 1

    def point_plot(algo):
        beta = 0.64
        for j in range(1):
            X_point = []
            step = 2
            for n in range(10,100,step):
                norm_w, particles, likelihood = algo(beta, y, n)
                point_x = sum(np.multiply(norm_w[:,T], particles[:,T]))
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
        T = 100
        beta = 0.64
        
        norm_w, particles, likelihood=log_SIS(beta, y, 100)
        X = []
        for i in range(T):
            x_1 = sum(np.multiply(norm_w[:,i], particles[:,i]))
            X.append(x_1)
        plt.plot(X, label = "log_SIS")

        norm_w, particles, likelihood=log_BPF(beta, y, 100, 'multi')
        X = []
        for i in range(T):
            x_1 = sum(np.multiply(norm_w[:,i], particles[:,i]))
            X.append(x_1)
        plt.plot(X, label = "log_BPF multi")

        norm_w, particles, likelihood=log_BPF(beta, y, 100, 'stratified')
        X = []
        for i in range(T):
            x_1 = sum(np.multiply(norm_w[:,i], particles[:,i]))
            X.append(x_1)
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
            norm_w, particles, likelihood = algo(beta, y, 100)
            list_beta.append(likelihood)
        plt.plot(I, list_beta)
        plt.show()

    def generateData(Q=1, R=1, x0=0, T=T):
        x = np.zeros(T)
        y = np.zeros(T)
        q = np.sqrt(Q)
        r = np.sqrt(R)
        x[0] = x0  # deterministic initial state
        y[0] = x[0] + r * np.random.normal(size=1)

        for t in range(1, T):
            x[t] = 0.75 * (x[t - 1]) + q * np.random.normal(size=1)
            y[t] = x[t] + r * np.random.normal(size=1)
        return x, y

    def test_PMH():
        """
        TO DO:
        - propagate x0 to gibbs
        - use the prior
        - why condition < 1
        """

        iterations = 1000
        beta_est, sigma_est, z = PMH(y, x0=1, phi=1, N=100, iterations=iterations)

        burnIn = 200#int(iterations * .3)
        # start = 1000
        # plt.hist(beta_est[burnIn:], bins=20, density=True, label="beta estimation")
        # plt.hist(sigma_est[burnIn:], bins=20, density=True, label="sigma estimation")
        # plt.legend(loc="upper left")
        # plt.xlabel("estimated value")
        # # plt.savefig('A_est.png')
        # plt.show()

        nBins = int(np.floor(np.sqrt(iterations - burnIn)))
        Lbeta = beta_est[burnIn:]
       # Plot the parameter posterior estimate (solid black line = posterior mean)
        plt.subplot(2, 1, 1)
        plt.hist(Lbeta, nBins, normed=1, facecolor='#1B9E77')
        plt.xlabel("beta^2")
        plt.ylabel("posterior density estimate")
        plt.axvline(np.mean(Lbeta), color='k')

        # Plot the trace of the Markov chain after burn-in (solid black line = posterior mean)
        plt.subplot(2, 1, 2)
        plt.plot(beta_est, color='#1B9E77')
        plt.xlabel("iteration")
        plt.ylabel("beta^2")
        plt.axhline(np.mean(Lbeta), color='k')
        plt.savefig('beta.png')
        # # Plot the autocorrelation function
        # plt.subplot(3, 1, 3)
        # macf = np.correlate(Lbeta - np.mean(Lbeta), Lbeta - np.mean(Lbeta), mode='full')
        # idx = int(macf.size / 2)
        # macf = macf[idx:]
        # macf = macf[0:100]
        # macf /= macf[0]
        # grid = range(len(macf))
        # plt.plot(grid, macf, color='#1B9E77')
        # plt.xlabel("lag")
        # plt.ylabel("ACF of beta^2")

        plt.show()

        LSigma = sigma_est[burnIn:, ]
        # Plot the parameter posterior estimate (solid black line = posterior mean)
        plt.subplot(2, 1, 1)
        plt.hist(LSigma, nBins, normed=1, facecolor='#1B1E22')
        plt.xlabel("sigma^2")
        plt.ylabel("posterior density estimate")
        plt.axvline(np.mean(LSigma), color='k')

        # Plot the trace of the Markov chain after burn-in (solid black line = posterior mean)
        plt.subplot(2, 1, 2)
        plt.plot(sigma_est, color='#1B1E22')
        plt.xlabel("iteration")
        plt.ylabel("sigma^2")
        plt.axhline(np.mean(LSigma), color='k')
        plt.savefig('sigma.png')
        # Plot the autocorrelation function
        # plt.subplot(3, 1, 3)
        # macf = np.correlate(LSigma - np.mean(LSigma), Lbeta - np.mean(LSigma), mode='full')
        # idx = int(macf.size / 2)
        # macf = macf[idx:]
        # macf = macf[0:100]
        # macf /= macf[0]
        # grid = range(len(macf))
        # plt.plot(grid, macf, color='#1B1E22')
        # plt.xlabel("lag")
        # plt.ylabel("ACF of sigma^2")

        plt.show()

    # algo = log_SIS
    #algo(0.64, y)
    #likelihood_plot(algo)
    #point_plot3()
    #weights_plot(algo)
    # point_plot(algo)
    test_PMH()

if __name__ == "__main__": main()
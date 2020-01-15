from math import *
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import normal
import DataGenerator
import scipy.stats as stats
from tqdm import tqdm

phi = 1.

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

def CSMC2(xref, y, beta, sigma, N):
    phi = 1
    T = len(y)

    logLikelihood = 0
    particles = np.zeros((N, T))
    normalisedWeights = np.zeros((N, T))
    B = np.zeros((N, T))

    # Init state, at t=0
    x0 = normal(0, sigma, N-1) 
    particles[:-1, 0] = x0
    particles[-1, 0] = xref[0]

    normalisedWeights[:, 0] = 1/N  # Save the normalized weights
    B[:, 0] = list(range(N))

    for t in range(1, T):

        newAncestors = multinomial_resampling(normalisedWeights[:,t-1]) 
        newAncestors[-1] = N-1 
        newAncestors = newAncestors.astype(int)
        B[:, t - 1] = newAncestors
        resample_particules = particles[:,t-1][newAncestors] 

        for i in range(N):
            particles[i, t] = normal(phi*resample_particules[i],sigma)
        particles[-1,t] = xref[t] 

        # weighting step
        logweights = np.zeros(N)
        for i in range(N):
            logweights[i] = stats.norm.logpdf(y[t], 0, beta*np.exp(particles[i, t]))
       
        max_weight = np.max(logweights)  
        w_p = np.exp(logweights - max_weight)
        normalisedWeights[:, t] = w_p / np.sum(w_p)  

        logLikelihood = logLikelihood + max_weight + np.log(np.sum(w_p)) - np.log(N) 

    B[:, T - 1] = list(range(N))
    return(normalisedWeights,particles, logLikelihood, B) 

def x_b(x, weights, B, T):
    x_star = np.zeros(T)
    J = np.where(np.random.uniform(size=1) < np.cumsum(weights[:, T - 1]))[0][0]

    for t in range(T):
        x_star[t] = x[int(B[J, t]), t]

    return(x_star)

def PG(xref, y, beta2Init, sigma2Init, prior_a, prior_b, N, M):
    # Number of states
    T = len(y)

    # Initialize the state parameters
    Lbeta2 = np.zeros(M)
    Lsigma2 = np.zeros(M)
    X = np.zeros((M, T))

    Lbeta2[0] = beta2Init
    Lsigma2[0] = sigma2Init
    x_ref = np.zeros(T)

    # Initialize the state by running a CPF
    beta2 = Lbeta2[0]
    sigma2 = Lsigma2[0]

    norm_w, particles, likelihood, B = CSMC2(xref, y, np.sqrt(beta2), np.sqrt(sigma2), N)
    X[0, :] = x_b(particles, norm_w, B, T)

    # Run MCMC loop
    for m in tqdm(range(1, M)):

        # Sample the parameters (inverse gamma posteriors)
        err_beta2 = np.zeros(T)
        err_sigma2 = np.zeros(T)
        for t in range(1,T):
            err_beta2[t] = np.exp(-X[m - 1, t])*y[t]**2
            err_sigma2[t] = X[m - 1, t] - phi*X[m - 1, t - 1]

        err_beta2 = np.sum(err_beta2)
        err_sigma2 = np.sum(err_sigma2 **2)

        Lbeta2[m] = stats.invgamma.rvs(a=prior_a + T/2, scale=prior_b + err_beta2 / 2, size=1)
        Lsigma2[m] = stats.invgamma.rvs(a=prior_a + T/2, scale=prior_b + err_sigma2 / 2, size=1)

        # Run CPF
        beta2 = Lbeta2[m]
        sigma2 = Lsigma2[m]
        #print("beta", np.sqrt(beta2))
        #print("sigma", np.sqrt(sigma2))
        norm_w, particles, likelihood, B = CSMC2(X[m - 1, :], y, np.sqrt(beta2), np.sqrt(sigma2), N)
        X[m, :] = x_b(particles, norm_w, B, T)

    return Lbeta2, Lsigma2

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


    N = 50
    M = 1100
    burnIn = 100

    Lbeta, Lsigma = PG(x, y, 1, 1, 0.01, 0.01, N,  M)
    grid = np.arange(burnIn, M, 1)
    nBins = int(np.floor(np.sqrt(M - burnIn)))

    Lbeta = Lbeta[burnIn:,]
    # Plot the parameter posterior estimate (solid black line = posterior mean)
    plt.subplot(3, 1, 1)
    plt.hist(Lbeta, nBins, normed=1, facecolor='#1B9E77')
    plt.xlabel("beta^2")
    plt.ylabel("posterior density estimate")
    plt.axvline(np.mean(Lbeta), color='k')

    # Plot the trace of the Markov chain after burn-in (solid black line = posterior mean)
    plt.subplot(3, 1, 2)
    plt.plot(Lbeta, color='#1B9E77')
    plt.xlabel("iteration")
    plt.ylabel("beta^2")
    plt.axhline(np.mean(Lbeta), color='k')

    # Plot the autocorrelation function
    plt.subplot(3, 1, 3)
    macf = np.correlate(Lbeta - np.mean(Lbeta), Lbeta - np.mean(Lbeta), mode='full')
    idx = int(macf.size / 2)
    macf = macf[idx:]
    macf = macf[0:100]
    macf /= macf[0]
    grid = range(len(macf))
    plt.plot(grid, macf, color='#1B9E77')
    plt.xlabel("lag")
    plt.ylabel("ACF of beta^2")

    plt.show()

    LSigma= Lsigma[burnIn:,]
    # Plot the parameter posterior estimate (solid black line = posterior mean)
    plt.subplot(3, 1, 1)
    plt.hist(LSigma, nBins, normed=1, facecolor='#1B1E22')
    plt.xlabel("sigma^2")
    plt.ylabel("posterior density estimate")
    plt.axvline(np.mean(LSigma), color='k')

    # Plot the trace of the Markov chain after burn-in (solid black line = posterior mean)
    plt.subplot(3, 1, 2)
    plt.plot(LSigma, color='#1B1E22')
    plt.xlabel("iteration")
    plt.ylabel("sigma^2")
    plt.axhline(np.mean(LSigma), color='k')

    # Plot the autocorrelation function
    plt.subplot(3, 1, 3)
    macf = np.correlate(LSigma - np.mean(LSigma), Lbeta - np.mean(LSigma), mode='full')
    idx = int(macf.size / 2)
    macf = macf[idx:]
    macf = macf[0:100]
    macf /= macf[0]
    grid = range(len(macf))
    plt.plot(grid, macf, color='#1B1E22')
    plt.xlabel("lag")
    plt.ylabel("ACF of sigma^2")

    plt.show()


if __name__ == "__main__": main()
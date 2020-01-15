from math import *
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import normal
import scipy.stats as stats
import sys
from scipy.stats import invgamma
from tqdm import tqdm


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


def log_BPF(phi, beta, sigma, y, N, sampling_method="multi"):
    # sigma = 0.16
    # phi = 1

    T = len(y)

    logLikelihood = 0
    particles = np.zeros((N, T))
    normalisedWeights = np.zeros((N, T))

    # Init state, at t=0
    x0 = normal(0, sigma, N)
    particles[:, 0] = x0

    logweights_0 = np.zeros(N)
    # for i in range(N):
    #    logweights_0[i] = np.log(stats.norm.pdf(y[0], 0, beta*np.exp(x0[i]/2)))
    logweights_0 = stats.norm.logpdf(y[0], 0, beta * np.exp(x0 / 2))

    max_weight_0 = np.max(logweights_0)  # Subtract the maximum value for numerical stability
    w_p_0 = np.exp(logweights_0 - max_weight_0)
    normalisedWeights[:, 0] = w_p_0 / np.sum(w_p_0)  # Save the normalized weights

    # accumulate the log-likelihood
    logLikelihood = logLikelihood + max_weight_0 + np.log(sum(w_p_0)) - np.log(N)

    for t in range(1, T):
        # Resampling
        if sampling_method == "multi":
            indexes = multinomial_resampling(normalisedWeights[:, t - 1])
        elif sampling_method == "stratified":
            indexes = stratified_resampling(normalisedWeights[:, t - 1])
        else:
            sys.exit('Specify a sampling method. Either "muli" or "stratified"')
        resample_particules = particles[:, t - 1][indexes]

        for i in range(N):
            particles[i, t] = normal(phi * resample_particules[i], sigma)

        # weighting step
        logweights = np.zeros(N)
        for i in range(N):
            # if stats.norm.pdf(y[t], 0, beta*np.exp(particles[i, t]))==0:
            #    break
            # else:
            logweights[i] = stats.norm.logpdf(y[t], 0, beta * np.exp(particles[i, t]))

        max_weight = np.max(logweights)  # Subtract the maximum value for numerical stability
        w_p = np.exp(logweights - max_weight)
        # w_p = np.multiply(normalisedWeights[:, t-1], w_p)
        normalisedWeights[:, t] = w_p / np.sum(w_p)  # Save the normalized weights

        logLikelihood = logLikelihood + max_weight + np.log(np.sum(w_p)) - np.log(N)

    return (normalisedWeights, particles, logLikelihood)


def PMH(y, x0, phi=1, N=100, iterations=100):
    # A = np.zeros(iterations)
    beta = np.zeros(iterations)
    sigma = np.zeros(iterations)
    z = np.zeros(iterations)
    step1 = 0.01

    # A[0] = .5
    beta[0] = .5
    sigma[0] = .1

    z[0] = log_BPF(phi, sqrt(beta[0]), sqrt(sigma[0]), y, N, sampling_method="multi")[2]

    for i in tqdm(range(1, iterations)):
        # Propose new parameters
        # A_proposed = A[i - 1] + step1 * np.random.normal(size=1)
        beta_proposed = beta[i - 1] + step1 * np.random.normal(size=1)
        sigma_proposed = sigma[i - 1] + step1 * np.random.normal(size=1)
        # while (abs(A_proposed) > 1):
        #     A_proposed = A[i - 1] + step1 * np.random.normal(size=1)
        while (beta_proposed > 1 or beta_proposed <= 0):
            beta_proposed = beta[i - 1] + step1 * np.random.normal(size=1)  # inv_gamma(.01,.01,x0)
        while (sigma_proposed > 1 or sigma_proposed <= 0):
            sigma_proposed = sigma[i - 1] + step1 * np.random.normal(size=1)  # inv_gamma(.01,.01,x0)

        # Run a PF to evaluate the likelihood
        # z_hat = log_BPF(y=y, A=A_proposed, Q=Q, R=R, x0=x0, N=N)
        z_hat = log_BPF(phi, sqrt(beta_proposed), sqrt(sigma_proposed), y, N, sampling_method="multi")[2]
        # Sample from uniform
        u = stats.uniform.rvs()

        # Compute the acceptance probability
        # numerator = z_hat + stats.norm.logpdf(A_proposed)
        numerator = z_hat + log(invgamma.pdf(a=.01, scale=.01, x=beta_proposed)) + log(
            invgamma.pdf(a=.01, scale=.01, x=sigma_proposed))
        # denominator = z[i - 1] + stats.norm.logpdf(A[i - 1])
        denominator = z[i - 1] + log(invgamma.pdf(a=.01, scale=.01, x=beta[i - 1])) + log(
            invgamma.pdf(a=.01, scale=.01, x=sigma[i - 1]))

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
            z[i] = z[i - 1]
            # A[i] = A[i-1]
            beta[i] = beta[i - 1]
            sigma[i] = sigma[i - 1]

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

    def test_PMH():
        """
        TO DO:
        - propagate x0 to gibbs
        - use the prior
        - why condition < 1
        """

        iterations = 1000
        beta_est, sigma_est, z = PMH(y, x0=1, phi=1, N=100, iterations=iterations)

        burnIn = 200  # int(iterations * .3)

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

        plt.show()

    test_PMH()


if __name__ == "__main__": main()
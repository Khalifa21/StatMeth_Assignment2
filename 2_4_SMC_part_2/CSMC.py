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

def CSMC(xref, y, beta, sigma, N):
    phi = 1
    T = len(y)

    logLikelihood = 0
    particles = np.zeros((N, T))
    normalisedWeights = np.zeros((N, T))
    B = np.zeros((N, T))

    # Init state, at t=0
    #x0 = normal(0, sigma, N-1) 
    #particles[:-1, 0] = x0
    #particles[-1, 0] = xref[0]

    x0 = normal(0, sigma)
    particles[:-1, 0] = x0

    normalisedWeights[:, 0] = 1/N  # Save the normalized weights
    B[:, 0] = list(range(N))

    #logweights_0 = np.zeros(N)
    #for i in range(N):
    #    logweights_0[i] = np.log(stats.norm.pdf(y[0], 0, beta*np.exp(particles[i, 0]/2)))

    #max_weight_0 = np.max(logweights_0)  # Subtract the maximum value for numerical stability
    #w_p_0 = np.exp(logweights_0 - max_weight_0)
    #normalisedWeights[:, 0] = w_p_0 / np.sum(w_p_0)  # Save the normalized weights

    # accumulate the log-likelihood
    #logLikelihood = logLikelihood + max_weight_0 + np.log(sum(w_p_0)) - np.log(N)

    for t in range(1, T):
        newAncestors = multinomial_resampling(normalisedWeights[:,t-1])
        resample_particules = particles[:,t-1][newAncestors]
        for i in range(N):
            particles[i, t] = normal(phi*resample_particules[i],sigma)

        # weighting step
        logweights = np.zeros(N)
        for i in range(N):
            logweights[i] = stats.norm.logpdf(y[t], 0, beta*np.exp(particles[i, t]))
       
        max_weight = np.max(logweights)  # Subtract the maximum value for numerical stability
        w_p = np.exp(logweights - max_weight)
        new_weights = np.exp(logweights - max_weight)
        w = new_weights / sum(new_weights)  # Save the normalized weights

        ancestors = multinomial_resampling(w)
        newAncestors = newAncestors[ancestors]
        newAncestors[N - 1] = N - 1

        #normalisedWeights[:, t] = w_p / np.sum(w_p)  # Save the normalized weights
        
        newAncestors = newAncestors.astype(int)
        B[:, t - 1] = newAncestors

        logLikelihood = logLikelihood + max_weight + np.log(np.sum(w)) - np.log(N)  ###

        # propogation step
        particles[:, t] = particles[:, t][newAncestors]
        particles[N - 1, t] = xref[t]

        # weighting step
        logweights = np.zeros(N)
        for i in range(N):
            logweights[i] = stats.norm.logpdf(y[t], 0, beta*np.exp(particles[i, t]))
        max_weight = max(logweights)  # Subtract the maximum value for numerical stability
        new_weights = np.exp(logweights - max_weight) / w[ancestors]
        normalisedWeights[:, t] = new_weights / sum(new_weights)

    B[:, T - 1] = list(range(N))
    return(normalisedWeights,particles, logLikelihood,B) 

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

    #logweights_0 = np.zeros(N)
    #for i in range(N):
    #    logweights_0[i] = np.log(stats.norm.pdf(y[0], 0, beta*np.exp(particles[i, 0]/2)))

    #max_weight_0 = np.max(logweights_0)  # Subtract the maximum value for numerical stability
    #w_p_0 = np.exp(logweights_0 - max_weight_0)
    #normalisedWeights[:, 0] = w_p_0 / np.sum(w_p_0)  # Save the normalized weights

    # accumulate the log-likelihood
    #logLikelihood = logLikelihood + max_weight_0 + np.log(sum(w_p_0)) - np.log(N)

    for t in range(1, T):

        newAncestors = multinomial_resampling(normalisedWeights[:,t-1]) #5.
        newAncestors[-1] = N-1 #5.
        newAncestors = newAncestors.astype(int)
        B[:, t - 1] = newAncestors
        resample_particules = particles[:,t-1][newAncestors] #5.

        for i in range(N):
            particles[i, t] = normal(phi*resample_particules[i],sigma) #6.
        particles[-1,t] = xref[t] #6.

        # weighting step
        logweights = np.zeros(N)
        for i in range(N):
            logweights[i] = stats.norm.logpdf(y[t], 0, beta*np.exp(particles[i, t]))
       
        max_weight = np.max(logweights)  # 6.
        w_p = np.exp(logweights - max_weight)
        normalisedWeights[:, t] = w_p / np.sum(w_p)  # 6.

        logLikelihood = logLikelihood + max_weight + np.log(np.sum(w_p)) - np.log(N)  ###

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

    def plot_csmc():
        T = 100
        beta = 0.64
        sigma = 0.16
        for j in range(3):
            norm_w, particles, likelihood, B = CSMC(x, y, beta, sigma, 100)
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

    algo = log_BPF
    #plot_csmc()
    #algo(0.64, y)
    #likelihood_plot(algo)
    #point_plot2(algo)
    #weights_plot(algo)
    #point_plot(algo)
    #point_plot3()
    #likelihood_plot2(algo)

    #I = [x/10 for x in range(2,20,1)]
    #list_beta = []
    #for beta in I:
    #    print("beta = ", beta)
    #    norm_w, particles, likelihood, B = CSMC2(x, y, beta, 0.16, 100)
    #    list_beta.append(likelihood)
    #plt.plot(I, list_beta)
    #plt.show()

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

    LSigma= Lsigma[burnIn:,]/100
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


    #plt.plot(Lbeta)
    #plt.xlabel("Number of iterations M")
    #plt.ylabel("beta^2")
    #plt.show()

    #plt.plot(Lsigma)
    #plt.xlabel("Number of iterations M")
    #plt.ylabel("sigma^2")
    #plt.show()

    #nBins = int(np.floor(np.sqrt(M - burnIn)))
    #plt.hist(Lbeta[burnIn:,], nBins)
    #plt.xlabel("beta^2")
    #plt.ylabel("posterior density estimate")
    #plt.show()

    #plt.hist(Lsigma[burnIn:,]/100, nBins)
    #plt.xlabel("sigma^2")
    #plt.ylabel("posterior density estimate")
    #plt.show()

if __name__ == "__main__": main()
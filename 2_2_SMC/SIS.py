from math import *
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import normal
import DataGenerator

T = 200
phi = 1.
sigma = 0.16
beta = 0.64
N = 51

def SIS(y,T,N,x0, phi, sigma, beta):
    """ N : nb particles, T : nb of samples"""
    particles = np.zeros((N, T))
    normalisedWeights = np.zeros((N, T))
    weights = np.zeros((N, T))

    # Init state
    particles[:, 0] = x0  # Deterministic initial condition
    w = np.zeros(N)
    for i in range(N):
        w[i] = normal(0,beta*exp(particles[i,0]/2))
    
    weights[:,0] = w
    normalisedWeights[:, 0] = w / sum(w)

    for t in range(1,T):
        w = np.zeros(N)
        for i in range(N):
            particles[i, t] = normal(phi*particles[i, t - 1],sigma)
            w[i] = normal(0,beta*exp(particles[i,t]/2))
    
        w = np.multiply(normalisedWeights[:, t-1], w)
        #w = np.multiply(weights[:, t-1], w)
        weights[:, t] = w
        normalisedWeights[:, t] = w / sum(w)

    #point estimate
    point_x = sum(np.multiply(normalisedWeights[:,-1], particles[:,-1]))

    #variance normalized weights
    var_weights = np.var(normalisedWeights[:,-1])

    return(normalisedWeights,particles, point_x, var_weights)

def log_SIS(y,T,N,x0, phi, sigma, beta):
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
        w = np.zeros(N)
        logweights = np.zeros(N)
        for i in range(N):
            sigma_p = beta*exp(particles[i,t]/2)
            particles[i, t] = normal(phi*particles[i, t - 1],sigma)
            #w[i] = normal(0,beta*exp(particles[i,t]/2))
            logweights[i] = -np.log(sigma_p*np.sqrt(2*np.pi))-1/2*(particles[i,t]/sigma_p)**2
        max_weight = max(logweights)  # Subtract the maximum value for numerical stability
        w_p = np.exp(logweights - max_weight)
        w_p = np.multiply(normalisedWeights[:, t-1], w_p)
        normalisedWeights[:, t] = w_p / sum(w_p)

    #point estimate
    point_x = sum(np.multiply(normalisedWeights[:,-1], particles[:,-1]))/N

    #variance normalized weights
    var_weights = np.var(normalisedWeights[:,-1])

    return(normalisedWeights,particles, point_x, var_weights)

def main():
    x0 = normal(0,sigma)
    x,y = DataGenerator.SVGenerator2(phi,sigma, beta,T)
    print("real x_T", x[-1])

    #def point_plot():
    #    for j in range(3):
    #        X_point = []
    #        for n in range(10,100,1):
    #            norm_w,particles,point_x, var_weights=SIS(y,T,n,x[0], phi, sigma, beta)
    #            X_point.append(1/2*(point_x-x[-1])**2)
    #        plt.plot(X_point)
    #        print(j, "done")
    #    plt.xlabel('number of samples N')
    #    plt.ylabel('mse')
    #    plt.show()

    def point_plot():
        X_point = []
        for n in range(10,200,1):
            norm_w,particles,point_x, var_weights=log_SIS(y,T,n,x0, phi, sigma, beta)
            X_point.append(1/2*(point_x-x[-1])**2)
            #print(point_x)
        plt.plot(X_point)
        plt.xlabel('number of samples N')
        plt.ylabel('mse')
        plt.show()

    def point_plot2():
        for j in range(1):
            norm_w,particles,point_x,var_weights=log_SIS(y,T,20,x0, phi, sigma, beta)
            X = []
            for i in range(T):
                x_1 = sum(np.multiply(norm_w[:,i], particles[:,i]))
                X.append(x_1)
            plt.plot(X, label = "estimated x "+str(j))
        plt.plot(x, label = "true x")
        plt.xlabel("T")
        plt.ylabel("xt")
        plt.legend()
        plt.show()

    def var_plot():
        X_point = []
        for n in range(10,100,1):
            norm_w,particles,point_x, var_weights=log_SIS(y,T,n,x0, phi, sigma, beta)
            X_point.append(var_weights)

        plt.plot(X_point)
        plt.xlabel("N")
        plt.ylabel("weight variance")
        plt.show()

    def weights_plot():
        norm_w,particles,point_x,var_weights=log_SIS(y,T,N,x0, phi, sigma, beta)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.hist(norm_w[:, 1])
        ax2.hist(norm_w[:, 10])
        ax3.hist(norm_w[:, 50])
        ax1.set_title('normalized weights iter 1')
        ax2.set_title('normalized weights iter 10')
        ax3.set_title('normalized weights iter 50')
        print(var_weights)
        plt.show()

    #weights_plot()
    #point_plot2()
    #var_plot()
    point_plot()

if __name__ == "__main__": main()

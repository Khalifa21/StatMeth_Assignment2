from math import *
from numpy.random import normal
import matplotlib.pyplot as plt

T = 100
phi = 1.
sigma = 0.16
beta = 0.64

def SVGenerator(phi, sigma, beta, T):
    xt=[None for i in range(T)]
    yt=[None for i in range(T)]

    xt[0]=normal(0,sigma**2)
    yt[0]=normal(0,beta*exp(xt[0]))

    for t in range(1,T):
        xt[t]=f(phi, sigma,xt,t-1)
        yt[t]=g(beta,xt[t])

    return xt,yt

def f(x,t):
    return(normal(phi*x[t],sigma**2))

def g(beta,x):
    return(normal(0,beta*exp(x)))
   

def SVGenerator2(phi, sigma, beta, T):
    xt=[None for i in range(T)]
    yt=[None for i in range(T)]

    xt[0]=normal(0,sigma**2)
    yt[0]=normal(0,beta*exp(xt[0]))

    for t in range(1,T):
        xt[t]=normal(phi*xt[t-1],sigma**2)
        yt[t]=normal(0,beta*exp(xt[t]))

    return xt,yt

def plot_xy(x,y,T):
    x_axis = list(range(1, T + 1))
    plt.plot(x_axis,x,label = "x")   
    plt.plot(x_axis,y, label = "y")
    plt.legend(loc="upper left")
    plt.show()

def main():
    x,y = SVGenerator2(phi,sigma, beta,T)
    plot_xy(x,y,T)

if __name__ == "__main__": main()
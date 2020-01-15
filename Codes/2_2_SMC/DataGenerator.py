from math import *
from numpy.random import normal
import matplotlib.pyplot as plt
import numpy as np

T = 50
phi = 1.
sigma = 0.16
beta = 0.64

   
def SVGenerator(phi, sigma, beta, T, saveOpt = True):
    xt=[None for i in range(T)]
    yt=[None for i in range(T)]

    xt[0]=normal(0,sigma)
    yt[0]=normal(0,beta*exp(xt[0]/2))

    for t in range(1,T):
        xt[t]=normal(phi*xt[t-1],sigma)
        yt[t]=normal(0,beta*exp(xt[t]/2))

    if saveOpt:
        np.savetxt("params.txt",np.array([phi,sigma,beta]),fmt='%.17f')
        np.savetxt("y.txt",np.array(yt),fmt='%.17f')
        np.savetxt("x.txt",np.array(xt),fmt='%.17f')

    return xt,yt

def plot_xy(x,y,T):
    x_axis = list(range(1, T + 1))
    plt.plot(x_axis,x,label = "x")   
    plt.plot(x_axis,y, label = "y")
    plt.legend(loc="upper left")
    plt.xlabel("T")
    plt.show()

def main():
    x,y = SVGenerator(phi,sigma, beta,T)
    plot_xy(x,y,T)

if __name__ == "__main__": main()

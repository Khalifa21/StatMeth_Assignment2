from math import *
from numpy.random import normal
import matplotlib.pyplot as plt
import numpy as np
import DataGenerator

T = 100
phi = 1.
sigma = 0.16
beta = 0.64

def main():
    x,y = DataGenerator.SVGenerator(phi,sigma, beta,T)
    #DataGenerator.plot_xy(x,y,T)

if __name__ == "__main__": main()
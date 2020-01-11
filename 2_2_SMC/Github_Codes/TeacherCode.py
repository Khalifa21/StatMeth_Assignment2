import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import scipy.stats as stats


# target distribution U[0,4]
def target_distn(x):
    #return  np.exp(-0.5* x**2) * ( 6*np.sin(6*x)**2 + 3*np.cos(x)**2 * np.sin(4*x)**2 + 2)
    return  0.1*np.exp(- 0.36*x**2) + 1.3*x**2*np.exp(-x**2)

X = np.linspace(-5, 5, 2000)
transformed_data = target_distn(X)


def proposal_distn(x):
    return stats.norm.pdf(x, loc=0, scale=1)

k = max(target_distn(X) / proposal_distn(X))
k=3
print(k)

def importance_sampling(iter=100):
    samples = np.zeros(iter)
    weights = np.zeros(iter)

    for i in range(iter):
        z = np.random.normal()
        samples[i] = z
        weights[i] = target_distn(z)/proposal_distn(z)
        if i in [01,10,50]:
            plt.hist(weights)

    return np.multiply(samples, weights)

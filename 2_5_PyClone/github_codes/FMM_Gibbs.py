from scipy import stats
import numpy as np
import numpy.random as npr
#import NonParamBayes.baseClassDP as dp

class BaseDP(object):
    def __init__(self, x, alpha, sigma, prior_mean=0, prior_var =1, nClass=3):
        """ """
        self.x = x
        self.sigma = sigma
        self.N = len(x)
        self.nClass = nClass # no of classes
        self.alpha = alpha
        self.pi = None
        self.z = None
        self.cluster_means = None
        self.prior_mean = prior_mean
        self.prior_var = prior_var

    def performGibbsSampling(self, iter):  # and register
        print("performGibbsSampling")

class FiniteMixtureModel(BaseDP):
    def __init__(self, x, alpha, sigma, prior_mean=0, prior_var =1, nClass=3):
        """ This method is called when you create an instance of the class."""
        BaseDP.__init__(self, x, alpha, sigma, prior_mean, prior_var, nClass)
        self.counts = np.zeros(nClass)
        self.cluster_means = np.zeros(nClass)

        self.initializePi(self.alpha)
        self.initialize_z()
        self.initialize_means()
        self.compute_suff_statistics()


    def initializePi(self, alpha):
        """
        sample pi from prior
        """
        param = np.ones(self.nClass)*alpha/self.nClass
        self.pi = stats.dirichlet(param).rvs(size=1).flatten()

    # class assignment
    def initialize_z(self):
        if self.pi is None:
            self.initializePi(self.alpha)

        self.z = np.zeros((self.N, self.nClass)).astype(int)
        for i in range(self.N):
            self.z[i, :] = npr.multinomial(1, self.pi).astype(int)

            # class assignment

    def initialize_means(self):
        """
        sample means from prior
        """
        self.cluster_means = npr.normal(loc=self.prior_mean, scale=np.sqrt(self.prior_var),
                                size=3)

    def sample_pi(self):
        """
        sample pi from posterior
        """
        param = np.ones(self.nClass) * self.alpha / self.nClass
        param += self.counts
        self.pi = stats.dirichlet(param).rvs(size=1).flatten()

    def sample_cluster_means(self):
        """
        sample cluster means from posterior
        """
        for i in range(self.nClass):
            numerator = (self.prior_mean / self.prior_var) + (self.cluster_means[i] * self.counts[i] / self.sigma**2)
            denominator = (1.0 / self.prior_var + self.counts[i] / self.sigma**2)
            posterior_mu = numerator / denominator
            posterior_var = 1.0 / denominator

            self.cluster_means[i] = npr.normal(loc=posterior_mu, scale=np.sqrt(posterior_var))


    def sample_z(self):
        """
        sample z matrix from posterior
        """
        for n in range(self.N):
            p = self.compute_pred_prob(n)
            self.z[n, :] = npr.multinomial(1, p).astype(int)

        #update sufficient statistics
        self.compute_suff_statistics()


    def compute_suff_statistics(self):
        """
        compute sufficient statistics
        """
        self.counts = np.sum(self.z, axis=0)
        for i in range(self.nClass):
            cls_index = self.z[:,i]>0
            self.cluster_means[i] = np.mean(self.x[cls_index], axis=0)

    def compute_pred_prob(self, data_index):
        """
        compute sufficient statistics
        """
        #data_cls_indx = np.flatnonzero(self.z[data_index, :])
        pred_prob = np.zeros(self.nClass)
        for cls_inx in range(self.nClass):
            pi = self.pi[cls_inx]
            loc = self.cluster_means[cls_inx]
            pred_prob[cls_inx] = pi * stats.norm.pdf(self.x[data_index], loc=loc, scale=self.sigma)

        return pred_prob/sum(pred_prob)

    def performGibbsSampling(self, iter=10):  # and register
        print("performGibbsSampling")
        for i in range(iter):
            self.sample_z()
            self.sample_pi()
            self.sample_cluster_means()
            self.plot_cluster_points()
            plt.show()
            plt.clf()

    def plot_cluster_hist(self):
        cls_data = []
        for i in range(self.nClass):
            cls_index = self.z[:, i] > 0
            cls_data.append(self.x[cls_index])

        plt.hist(cls_data,
                 bins=20,
                 histtype='stepfilled', alpha=.5)

        plt.show()

    def plot_cluster_points(self):
        cls_data = []
        mark_style = [".", "+", "*"]
        mark_col = ["red", "green", "blue"]
        for i in range(self.nClass):
            cls_index = self.z[:, i] > 0
            cls_data.append([self.x[cls_index]])
            y = np.ones(len(cls_data[i][0]))*0
            plt.scatter(cls_data[i], y, marker=mark_style[i], color=mark_col[i])
        #plt.pause(0.1)
        #plt.show()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    def generate_data(pi, means, sigma, N=10, nClass=3):
        """Generate data
        """
        x = np.zeros(N)
        y = np.zeros((N, nClass)).astype(int)
        for i in range(N):
            y[i, :] = npr.multinomial(1, pi).astype(
                int)  # np.random.choice(np.array([1,2,3]), p=pi) #npr.multinomial(1, pi)
            cls_index = np.flatnonzero(y[i])[0]
            x[i] = npr.normal(loc=means[cls_index], scale=sigma, size=1)
        return x


    data = generate_data([.3, .5, .2], [-3, 0, 3], .5, N=20)
    #plt.hist(data)
    #plt.show()

    fmm = FiniteMixtureModel(data, alpha=10, sigma=.5, prior_mean=0, prior_var =1, nClass=3)
    fmm.plot_cluster_points()
    fmm.performGibbsSampling()
    '''
    # plot clusters before running the Gibbs sampler
    fmm.plot_cluster_hist()
    fmm.performGibbsSampling()
    # plot clusters after running the Gibbs sampler
    fmm.plot_cluster_hist()
    '''
    #print(fmm.counts)
    #print(fmm.means)

from scipy import stats
import numpy as np
import numpy.random as npr


class BaseDP(object):
    def __init__(self, x, alpha, sigma2, prior_mean=0, prior_var =1, nClass=3):
        """ """
        self.x = x
        self.sigma2 = sigma2
        self.N = len(x)
        self.nClass = nClass # no of classes
        self.alpha = alpha
        self.z = None
        self.cluster_means = np.zeros(self.nClass)
        self.post_cluster_means = np.zeros(self.nClass)
        self.post_cluster_vars = np.zeros(self.nClass)
        self.prior_mean = prior_mean
        self.prior_var = prior_var

    def performGibbsSampling(self, iter):  # and register
        print("performGibbsSampling")

class FiniteMixtureModel(BaseDP):
    def __init__(self, x, alpha, sigma2, prior_mean=0, prior_var =1, nClass=3):
        """ This method is called when you create an instance of the class."""
        BaseDP.__init__(self, x, alpha, sigma2, prior_mean, prior_var, nClass)
        self.counts = np.zeros(nClass)

        self.initialize_z()
        self.compute_suff_statistics()

        self.plot_type = None


    # class assignment
    def initialize_z(self):
        # randomly sample from 1/3, 1/3, 1/3
        pi = [1/3, 1/3, 1/3]
        self.z = np.zeros((self.N, self.nClass)).astype(int)
        for i in range(self.N):
            self.z[i, :] = npr.multinomial(1, pi).astype(int)

            # class assignment


    def set_plot_type(self, plot_type):
        self.plot_type = plot_type


    def compute_cluster_param(self):
        """
        sample cluster means from posterior
        """
        for i in range(self.nClass):
            numerator = (self.prior_mean / self.prior_var) + (self.cluster_means[i] * self.counts[i] / self.sigma2)
            denominator = (1.0 / self.prior_var) + (self.counts[i] / self.sigma2)

            self.post_cluster_means[i] = numerator / denominator
            self.post_cluster_vars[i] = 1.0 / denominator

    def sample_z(self):
        """
        sample z matrix from posterior
        """
        for n in range(self.N):
            # suff stats after removing nth data point
            self.compute_suff_statistics(data_index=n)

            p = self.compute_pred_prob(n)
            self.z[n, :] = npr.multinomial(1, p).astype(int)

        #update sufficient statistics
        self.compute_suff_statistics()


    def compute_suff_statistics(self, data_index=None):
        """
        compute sufficient statistics
        """
        if data_index is None:
            z = self.z
            x = self.x
        else:
            z = np.delete(self.z, data_index, 0)
            x = np.delete(self.x, data_index, 0)

        self.counts = np.sum(z, axis=0)
        for i in range(self.nClass):
            cls_index = z[:,i]>0
            self.cluster_means[i] = np.mean(x[cls_index])

        self.compute_cluster_param()

    def compute_pred_prob(self, data_index):
        """
        compute sufficient statistics
        """
        #data_cls_indx = np.flatnonzero(self.z[data_index, :])
        pred_prob = np.zeros(self.nClass)
        for cls_inx in range(self.nClass):
            pi = (self.counts[cls_inx] + self.alpha/self.nClass)#/(self.N-1+self.alpha)
            loc = self.post_cluster_means[cls_inx]
            scale = self.post_cluster_vars[cls_inx] + self.sigma2
            pred_prob[cls_inx] = pi * stats.norm.pdf(self.x[data_index], loc=loc, scale=np.sqrt(scale))

        return pred_prob/sum(pred_prob)

    def performGibbsSampling(self, iter=10):  # and register
        print("performGibbsSampling")
        for i in range(iter):
            self.sample_z()


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
    def generate_data(pi, means, sigma2, N=10, nClass=3):
        """Generate data
        """
        x = np.zeros(N)
        y = np.zeros((N, nClass)).astype(int)
        for i in range(N):
            y[i, :] = npr.multinomial(1, pi).astype(
                int)  # np.random.choice(np.array([1,2,3]), p=pi) #npr.multinomial(1, pi)
            cls_index = np.flatnonzero(y[i])[0]
            x[i] = npr.normal(loc=means[cls_index], scale=np.sqrt(sigma2), size=1)
        return x


    npr.seed(11)
    data = generate_data([.3, .5, .2], [-3, 0, 3], sigma2=.5, N=30)
    #plt.hist(data)
    #plt.show()


    fmm = FiniteMixtureModel(data, alpha=.1, sigma2=.1, prior_mean=0, prior_var =1, nClass=3)
    fmm.plot_cluster_hist()
    plt.clf()
    fmm.performGibbsSampling(100)
    fmm.plot_cluster_points()
    plt.show()
    '''
    # plot clusters before running the Gibbs sampler
    fmm.plot_cluster_hist()
    fmm.performGibbsSampling()
    # plot clusters after running the Gibbs sampler
    fmm.plot_cluster_hist()
    '''

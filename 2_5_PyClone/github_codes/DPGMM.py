from scipy import stats
import numpy as np
import numpy.random as npr


class BaseDP(object):
    def __init__(self, x, alpha, sigma2, prior_mean=0, prior_var =1):
        """ """
        self.x = x
        self.sigma2 = sigma2
        self.N = len(x)
        self.nClass = None # no of classes
        self.alpha = alpha
        self.z = None
        self.prior_mean = prior_mean
        self.prior_var = prior_var

    def performGibbsSampling(self, iter):  # and register
        print("performGibbsSampling")

class DPMM(BaseDP):
    def __init__(self, x, alpha, sigma2, prior_mean=0, prior_var =1):
        """ This method is called when you create an instance of the class."""
        BaseDP.__init__(self, x, alpha, sigma2, prior_mean, prior_var)
        self.initialize_z()
        self.counts = np.zeros(self.nClass)

        self.post_cluster_means = np.zeros(self.nClass)
        self.post_cluster_vars = np.zeros(self.nClass)

        self.initialize_means()
        self.compute_suff_statistics()
        self.plot_type = None

    def DP_prior(self, alpha):
        """
        DP prior
        """
        z = np.zeros(self.N).reshape(-1,1)
        z[0] = 1
        for n in range(1, self.N):
            k = z.shape[1]
            prob = np.zeros(k+1)
            prob[0:k] = np.sum(z, axis=0)
            prob[k] = self.alpha
            prob = prob/sum(prob)
            cls_index = npr.choice(list(range(k+1)), p=prob)
            if cls_index == k:
                # new class assignment
                z = np.column_stack((z, np.zeros(self.N)))

            z[n, cls_index] = 1

        self.z = z
        self.nClass = z.shape[1]

    # class assignment
    def initialize_z(self):
        self.DP_prior(self.alpha)

            # class assignment

    def initialize_means(self):
        """
        sample means from prior
        """
        self.cluster_means = npr.normal(loc=self.prior_mean, scale=np.sqrt(self.prior_var),
                                size=3)

    def set_plot_type(self, plot_type):
        self.plot_type = plot_type

    def compute_cluster_param(self):
        """
        sample cluster means from posterior
        """
        self.post_cluster_means = np.zeros(self.nClass)
        self.post_cluster_vars = np.zeros(self.nClass)
        for i in range(self.nClass):
            numerator = (self.prior_mean / self.prior_var) + (self.cluster_means[i] * self.counts[i] / self.sigma2)
            denominator = (1.0 / self.prior_var + self.counts[i] / self.sigma2)

            self.post_cluster_means[i] = numerator / denominator
            self.post_cluster_vars[i] = 1.0 / denominator

    def sample_z(self):
        """
        sample z matrix from posterior
        """
        for n in range(self.N):
            # suff stats after removing nth data point
            self.compute_suff_statistics(data_index=n)

            prob = self.compute_pred_prob(n)

            cls_index = npr.choice(list(range(len(prob))), p=prob)
            z = self.z
            if cls_index == len(prob)-1:
                # create new class
                z = np.column_stack((self.z, np.zeros(self.N)))

            z[n, cls_index] = 1
            # remove zero columns/ empty cluster
            columncounts = np.sum(z, axis=0)
            nzc = np.where(columncounts > 0)[0]
            z = z[:, nzc]
            self.z = z
            self.nClass = z.shape[1]

        self.compute_suff_statistics()



    def compute_suff_statistics(self, data_index=None):
        """
        compute sufficient statistics
        """
        if data_index is not None:
            # remove nth point from the cluster
            # set cluster assignment to zero
            self.z[data_index,:] = 0
            # remove zero columns/ empty cluster
            columncounts = np.sum(self.z, axis=0)
            nzc = np.where(columncounts > 0)[0]
            self.z = self.z[:, nzc]
            self.nClass = self.z.shape[1]


        self.counts = np.sum(self.z, axis=0)
        self.cluster_means = np.zeros(self.nClass)
        for i in range(self.nClass):
            cls_index = self.z[:,i]>0
            self.cluster_means[i] = np.mean(self.x[cls_index])
        #update cluster parameters
        self.compute_cluster_param()


    def compute_pred_prob(self, data_index):
        pred_prob = np.zeros(self.nClass+1)
        for cls_inx in range(self.nClass):
            pi = self.counts[cls_inx]
            loc = self.post_cluster_means[cls_inx]
            scale = self.post_cluster_vars[cls_inx] + self.sigma2
            pred_prob[cls_inx] = np.log(pi) + stats.norm.logpdf(self.x[data_index], loc=loc, scale=np.sqrt(scale))

        pred_prob[self.nClass] = np.log(self.alpha) + stats.norm.logpdf(
                                    self.x[data_index],
                                    loc=self.prior_mean, scale=np.sqrt(self.prior_var))

        pred_prob = pred_prob - max(pred_prob)
        pred_prob = np.exp(pred_prob)

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
            try:
                plt.scatter(cls_data[i], y, marker=mark_style[i], color=mark_col[i])
            except:
                # for other classes use pink colour
                plt.scatter(cls_data[i], y, marker=".", color="pink")

        plt.show()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    def generate_data(pi, means, sigma2, N=10, nClass=3):
        """Generate data
        """
        x = np.zeros(N)
        z = np.zeros((N, nClass)).astype(int)
        for i in range(N):
            z[i, :] = npr.multinomial(1, pi).astype(int)
            cls_index = np.flatnonzero(z[i])[0]
            x[i] = npr.normal(loc=means[cls_index], scale=np.sqrt(sigma2), size=1)
        #print(z)
        return x


    data = generate_data([.3, .5, .2], [-3, 0, 3], .5, N=30)
    plt.hist(data)
    plt.show()
    npr.seed(22)
    fmm = DPMM(data, alpha=.1, sigma2=.5, prior_mean=0, prior_var=1)
    fmm.set_plot_type('scatter')

    fmm.plot_cluster_points()
    fmm.performGibbsSampling(iter=100)
    fmm.plot_cluster_points()
    print(fmm.nClass)

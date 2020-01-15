import numpy as np
import numpy.random as npr

def CRP(alpha, N):
    z = np.zeros(N).reshape(-1,1)
    z[0] = 1
    for n in range(1, N):
        k = z.shape[1]
        prob = np.zeros(k+1)
        prob[0:k] = np.sum(z, axis=0)
        prob[k] = alpha
        prob = prob/sum(prob)
        cls_index = npr.choice(list(range(k+1)), p=prob)
        if cls_index == k:
            # new class assignment
            z = np.column_stack((z, np.zeros(N)))

        z[n, cls_index] = 1

    return z

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.pylab as plt

    npr.seed(11122019)
    z = CRP(1, 15)
    print(z)
    with sns.axes_style("dark"):
        ax = sns.heatmap(z, linewidth=.1, cmap=plt.cm.cubehelix_r, annot=True, linecolor="black", cbar=False)
        plt.title(" ")
        plt.xlabel(" ")
        plt.ylabel(" ")
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.show()

    #plt.imshow(z, cmap='cool', interpolation='nearest')
    #plt.show()
    #print(z)

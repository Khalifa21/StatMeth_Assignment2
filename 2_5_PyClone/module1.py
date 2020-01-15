import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import scipy as sp

# Option with fixed number of clusters
class GEM:
    
    def __init__(self,alpha,cluster_number=None):
        self.alpha=alpha
        self.beta=[]
        
        self.pi=[]
        self.pi.append(0)
        
        self.cluster_number=cluster_number

        if self.cluster_number is None:
            self._add_stick()
        else:
            for i in range(cluster_number):
                self._add_stick()
        
            
    def _add_stick(self):
        self.beta.append(st.beta.rvs(1,self.alpha))
        self.pi.append( self.beta[-1]*(1-np.sum(self.pi)) )
    
    def _sample_pi(self,u):
        pi_prev = self.pi[0]
        pi_next = self.pi[1]
           
        k = 1
        while True:
            if pi_prev <= u and u < pi_next:
                    # if the random number is in this interval
                    # return the appropriate cluster
                    return k
            else:
                if self.cluster_number is None:
                    # Mode one: Infinite number of clusters
                    # ... otherwise go to next cluster
                    k+=1
                    pi_prev = pi_next
                    # if there are no more clusters in list, generate the next one...
                    if len(self.pi) is k:
                        self._add_stick()
                    #print(len(self.pi))
                    pi_next += self.pi[k]
                else:
                    # Mode two: Finite number of clusters
                    k+=1
                    if k > self.cluster_number: # if the next
                        return self.cluster_number-1
                    pi_prev = pi_next
                    pi_next += self.pi[k]
                    
        
        
    def sample(self,N):
        uni = st.uniform.rvs(size=N)
        return np.array([ self._sample_pi(u) for u in uni])

class DataGenerator:
    
    def __init__(self,alpha,cluster_number=None,max_d=1,fixed_trials=False):
        self.gem = GEM(alpha,cluster_number)
        self.max_d=max_d
        self.fixed_trials = fixed_trials
        
    def generate_data(self,N):
        Z = self.gem.sample(N)
        Z_map = {z:i for i,z in enumerate(set(Z))}
        
        phi = np.random.beta(1,1,len(Z_map))
        
        if self.fixed_trials:
            d = np.array([self.max_d for _ in range(N)])
        else:
            d = np.random.randint(1,self.max_d,size=[N])
            
        b = np.array( [ np.random.binomial(d[n],phi[ Z_map[Z[n]] ]) for n in range(N)] )
        
        return np.squeeze(b),np.squeeze(d),np.squeeze(Z),Z_map,np.squeeze(phi)

alpha = 1
num_of_clusters = 5
max_d = 100
fixed_trials = False 

dg = DataGenerator(alpha,num_of_clusters,max_d=max_d,fixed_trials=fixed_trials)
b,d,Z,Z_map,phi = dg.generate_data(100)

Z = map(lambda Zn : Z_map[Zn], Z)

data = np.array([ [bi,di,zi] for bi,di,zi in zip(b,d,Z) ])

plt.figure(figsize=(20,20))
for c in set(data[:,2]):
    data_c = data[data[:,2] == c]
    plt.scatter(data_c[:,0],data_c[:,1],label='cluster={}'.format(c))
    
plt.xlabel('$b_i$')
plt.ylabel('$d_i$')
plt.title('$b_i$/$d_i$ plot')
plt.legend()
    
plt.show()
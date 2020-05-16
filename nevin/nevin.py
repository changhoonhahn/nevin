'''



'''
import numpy as np 

from scipy.linalg import sqrtm
from scipy.special import digamma
from sklearn.neighbors import NearestNeighbors



def log_evidence(chain, lnlike_chain, prior_samples): 
    ''' calculate log evidence (aka Bayes Factor) 

    ln p(d|m) = int p(theta|d,m) ln p(d|theta,m) dtheta - D_KL( p(theta|d,m) || p(theta) )

    
    :param chain: 
        N x Ndim dimensional array of the MC chain where N is the number of
        samples, Ndim is the dimensions of your parameter space 

    :param lnlike_chain: 
        N dimensional array of log(likelihood) values for the chain.
        **Normalization of the log likelihood matters in this case!**

    :param prior_samples: 
        M dimensional array. M samples from the prior distribution. 

    :return log_evidence: 
        log(evidence) 

    
    notes:
    -----
    * currently implementation only includes Wang+(2009) KL divergence estimator. 
    * for flat or gaussian priors, we can use a hybrid divergence estimator
      and analytically evaluate q. This may improve stability (see
      https://github.com/changhoonhahn/nonGaussLike/blob/ed099aef531dfa0240626e7bd51d071a17ae47f1/notebook/local_KL_Xq.ipynb) 

    todo: 
    ----
    * [ ] include Pozcos+(2012) estimators 
    * [ ] warnings based on sample size and dimensionality
    * [ ] warnings/guidelines of choice of k 

    references:
    -----------
    * Trotta (2008)
    * Q. Wang, S. Kulkarni, & S. Verdu (2009): Divergence Estimation for
      Multidimensional Densities Via k-Nearest-Neighbor Distances. IEEE
      Transactions on Information Theory, 55(5), 2392-2405.i 

    '''
    N, dim = chain.shape
    M = prior_samples.shape[0]
    assert len(lnlike_chain) == N
    assert prior_samples.shape[1] == dim 

    # put some warnings here on sample size given the dimensions of the
    # parameter space once we have some intuition 

    # caluclate < ln p(d|theta,m) >, the posterior mean of the log likelihood 
    exp_lnlike = 1./float(N) * np.sum(lnlike_chain)

    # calculate KL divergence between the posterior and prior from samples
    # drawn from them using k-th Nearest Neighbor estimator 
    D_kl = KL_w2009_eq29(chain, prior_samples)

    return exp_lnlike + D_kl 


def sample_prior(priorfn, M): 
    ''' sample the prior M times using the prior function

    :param priorfn: 
        function of the prior.  

    :param M: 
        int. number of samples to draw from the prior distribution 
    '''
    return samples  


def KL_w2009_eq29(X, Y):
    ''' kNN KL divergence estimate using Eq. 29 from Wang et al. (2009). 
    This has some bias reduction applied to it and a correction for 
    epsilon.

    sources 
    ------- 
    - Q. Wang, S. Kulkarni, & S. Verdu (2009). Divergence Estimation for Multidimensional Densities Via k-Nearest-Neighbor Distances. IEEE Transactions on Information Theory, 55(5), 2392-2405.
    '''
    assert X.shape[1] == Y.shape[1]
    n, d = X.shape # X sample size, dimensions
    m = Y.shape[0] # Y sample size

    # first determine epsilon(i)
    NN_X = NearestNeighbors(n_neighbors=1).fit(X)
    NN_Y = NearestNeighbors(n_neighbors=1).fit(Y)
    dNN1_XX, _ = NN_X.kneighbors(X, n_neighbors=2)
    dNN1_XY, _ = NN_Y.kneighbors(X)
    eps = np.amax([dNN1_XX[:,1], dNN1_XY[:,0]], axis=0) * 1.000001

    # find l_i and k_i
    _, i_l = NN_X.radius_neighbors(X, eps)
    _, i_k = NN_Y.radius_neighbors(X, eps)
    l_i = np.array([len(il)-1 for il in i_l])
    k_i = np.array([len(ik) for ik in i_k])
    #assert l_i.min() > 0
    #assert k_i.min() > 0

    rho_i = np.empty(n, dtype=float)
    nu_i = np.empty(n, dtype=float)
    for i in range(n):
        rho_ii, _ = NN_X.kneighbors(np.atleast_2d(X[i]), n_neighbors=l_i[i]+1)
        nu_ii, _ = NN_Y.kneighbors(np.atleast_2d(X[i]), n_neighbors=k_i[i])
        rho_i[i] = rho_ii[0][-1]
        nu_i[i] = nu_ii[0][-1]

    d_corr = float(d) / float(n) * np.sum(np.log(nu_i/rho_i))
    return d_corr + np.sum(digamma(l_i) - digamma(k_i)) / float(n) + np.log(float(m)/float(n-1))

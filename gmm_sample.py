import numpy as np
import matplotlib.pyplot as plt

# arrays have shapes
#  mu is n-by-k-by-2
#  sigma is n-by-k-by-2
#  rho is n-by-k
#  pi is n-by-k
# def gmm_sample(mu, sigma, rho, pi, eos):

#     ##################################################
#     # verify shapes
    
#     n, k = rho.shape
#     assert mu.shape == (n, k, 2)
#     assert sigma.shape == (n, k, 2)
#     assert pi.shape == (n, k)
#     assert eos.shape == (n,)

#     ##################################################
#     # choose a mixture component
#     c = np.cumsum(pi, axis=1)
#     r = np.random.random((n,1))

#     # first index where r less than or equal to c
#     mixture_comp = (r <= c).argmax(axis=1)

#     ##################################################
#     # get that component of mu, sigma, rho for each item
    
#     idx = np.arange(n)
    
#     mu = mu[idx, mixture_comp]
#     sigma = sigma[idx, mixture_comp]
#     rho = rho[idx, mixture_comp].reshape((-1, 1))
    
#     ##################################################
#     # do sampling
    
#     s1 = sigma[:, 0].reshape((-1, 1))
#     s2 = sigma[:, 1].reshape((-1, 1))

#     a = s1
#     b = rho*s2
#     c = s2*np.sqrt(1.0 - rho**2)

#     rx = np.random.normal(size=(n, 1))
#     ry = np.random.normal(size=(n, 1))

#     rx_prime = a * rx
#     ry_prime = b * rx + c * ry

#     r = np.hstack((rx_prime, ry_prime)) + mu

#     eos_samples = (np.random.random((n,1)) <= eos).astype(np.float32)

#     return np.hstack((r, eos_samples))

def gmm_sample(mu, sigma, rho, pi, eos):

    ##################################################
    # verify shapes
    
    n, k = rho.shape
    assert mu.shape == (n, k, 2)
    assert sigma.shape == (n, k, 2)
    assert pi.shape == (n, k)
    assert eos.shape == (n,)

    ##################################################
    # choose a mixture component
    c = np.cumsum(pi, axis=1)
    r = np.random.random((n,1))

    # first index where r less than or equal to c
    mixture_comp = (r <= c).argmax(axis=1)

    ##################################################
    # get that component of mu, sigma, rho for each item
    
    idx = np.arange(n)
    
    mu = mu[idx, mixture_comp]
    sigma = sigma[idx, mixture_comp]
    rho = rho[idx, mixture_comp].reshape((-1, 1))
    
    ##################################################
    # do sampling
    
    s1 = sigma[:, 0].reshape((-1, 1))
    s2 = sigma[:, 1].reshape((-1, 1))
    mean = [mu[0,0], mu[0,1]]
    cov = [[s1*s1, rho*s1*s2], [rho*s1*s2, s2*s2]]
    x = np.random.multivariate_normal(mean, cov, 1)
    eos_samples = (np.random.random((n,1)) <= eos).astype(np.float32)
    return np.hstack((x, eos_samples))

    a = s1
    b = rho*s2
    c = s2*np.sqrt(1.0 - rho**2)

    rx = np.random.normal(size=(n, 1))
    ry = np.random.normal(size=(n, 1))

    rx_prime = a * rx
    ry_prime = b * rx + c * ry

    r = np.hstack((rx_prime, ry_prime)) + mu

    return np.hstack((r, eos_samples))

# arrays have shapes
#   mu is k-by-2
#   sigma is k-by-2
#   rho is (k,)
#   pi is (k,)
#   eos is a scalar
#   x is n-by-3 (x, y, eos)
def gmm_eval(mu, sigma, rho, pi, eos, x):

    ##################################################
    # verify shapes
    k = rho.shape[0]
    n = x.shape[0]
    assert mu.shape == (k, 2)
    assert sigma.shape == (k, 2)
    assert pi.shape == (k,)
    assert x.shape == (n, 3)
    assert np.isscalar(eos)

    ##################################################
    # do stuff from Graves paper to evaluate probs.
    
    dev = x[:, None, :2] - mu[None, :, :]
    var = sigma**2
    s1s2 = sigma.prod(axis=1)

    z12_before = dev**2 / var
    z12 = z12_before.sum(axis=2)

    reduce_dev = dev.prod(axis=2)
    z3 = (reduce_dev*rho*2) / (s1s2)
    Z = z12 - z3

    normalizer = (2.0 * np.pi * s1s2) * np.sqrt(1.0-rho**2)

    expon_part = np.exp(-Z / (2.0 * (1.0 - rho**2)))

    N = expon_part / normalizer

    p = (N * pi).sum(axis=1)

    x3 = x[:,2]
    p_eos = eos * x3 + (1.0-eos) * (1.0-x3)

    return p * p_eos

# ######################################################################

# def main():

#     mu = np.array([ 0.3, 0.4, -0.1, -0.7, -0.6, 0.8 ]).reshape((3, 2))
#     sigma = np.array([ 0.5, 0.2, 0.6, 0.5, 0.3, 0.3 ]).reshape((3, 2))
#     rho = np.array([ 0.7, -0.9, 0.1 ])   
#     pi = np.array([ 0.6, 0.3, 0.1 ])
#     eos = 0.05

#     rng = np.linspace(-2.0, 2.0, 100)
#     x, y = np.meshgrid(rng, rng)

#     m = np.hstack((x.reshape(-1, 1),
#                    y.reshape(-1, 1),
#                    np.zeros_like(y.reshape(-1, 1))))

#     p = gmm_eval(mu, sigma, rho, pi, eos, m)
#     p = p.reshape(x.shape)

#     # generate a number of samples of each thing
#     n_samples = 200

#     mu = np.tile(mu, (n_samples, 1, 1))
#     sigma = np.tile(sigma, (n_samples, 1, 1))
#     rho = np.tile(rho, (n_samples, 1))
#     pi = np.tile(pi, (n_samples, 1))
#     eos = np.tile(eos, (n_samples,))

#     samples = gmm_sample(mu, sigma, rho, pi, eos)
#     print('sampled eos at rate {}'.format(samples[:,2].mean()))

#     plt.pcolormesh(rng, rng, p)
#     plt.plot(samples[:,0], samples[:,1], 'k.')
#     plt.axis([-2, 2, -2, 2])
#     plt.show()

# if __name__ == '__main__':
#     main()









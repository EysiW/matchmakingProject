import numpy as np
import scipy as sp
import scipy.stats
import matplotlib.pyplot as plt
import numpy.random as rng
import math

my_s1 = 25
my_s2 = 25
var_s1 = 25/3
var_s2 = 25/3
var_t = 25/3
var_T = var_s1 + var_s2 + var_t


t_guess = 2 #unknown t in mean vector

my_S = [my_s1, my_s2]
sigma_s = [[var_s1, 0], [0, var_s2]]
sigma_t = var_t

# Samples
num_samples = 1000
s_sample = np.zeros((num_samples, 2))
t_sample = np.zeros(num_samples)

# Task specifik covariance matrix
det_T = 1/((1/var_s1+1/var_t)*(1/var_s2+1/var_t)-1/var_t**2)

Sigma_T = det_T*np.array([[1/var_s2 + 1/var_t, 1/var_t], [1/var_t, 1/var_s1 + 1/var_t]])

my_t = Sigma_T@np.array([[my_s1/var_s1 + t_guess/var_t], [my_s2/var_s1 - t_guess/var_t]])


# Gibbs Sampler
s_sample[0] = rng.multivariate_normal(my_t.T[0], sigma_s)
t_sample[0] = t_guess

for i in range(num_samples):
    my_t = Sigma_T@np.array([[my_s1/var_s1 + t_sample[i-1]/var_t], [my_s2/var_s1 - t_sample[i-1]/var_t]])
    s_sample[i] = rng.multivariate_normal(my_t.T[0], Sigma_T)
    t_sample[i] = sp.stats.truncnorm.rvs(0, math.inf, s_sample[i-1][0]-s_sample[i-1][1], sigma_t)


plt.hist(s_sample, bins=50)
plt.show()

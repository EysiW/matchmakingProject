import numpy as np
import scipy as sp
import scipy.stats
import matplotlib.pyplot as plt
import numpy.random as rng
import math
import pandas as pd

# Hyperparameters
my_s1 = 25
my_s2 = 25
var_s1 = 25/3
var_s2 = 25/3
var_t = 25/3
var_T = var_s1 + var_s2 + var_t

t_guess = 100 #unknown t in mean vector, init guess

# 2D case
my_S = [my_s1, my_s2]
sigma_s = [[var_s1, 0], [0, var_s2]]
sigma_t = var_t

# Samples
num_samples = 10200
s_sample = np.zeros((num_samples, 2))
t_sample = np.zeros(num_samples)

# Task specifik covariance matrix
det_T = 1/((1/var_s1+1/var_t)*(1/var_s2+1/var_t)-1/var_t**2)

# Gibbs Sampler
# Calculate my using corolary 1 
Sigma_T = det_T*np.array([[1/var_s2 + 1/var_t, 1/var_t], [1/var_t, 1/var_s1 + 1/var_t]])
def update_my(t, my_s1, var_s1, my_s2, var_s2, var_t, Sigma_T):
    return Sigma_T@np.array([[my_s1/var_s1 + t/var_t], [my_s2/var_s1 - t/var_t]])

# Gibbs Sampler function
# s_sample and t_ sample are two/one dimentional vectors storing the values calculated 
def gibbs_sample(t_guess, my_s1, var_s1, my_s2, var_s2, var_t, sigma_t, Sigma_T, s_sample, t_sample, y, num_samples):
    my = update_my(t_guess, my_s1, var_s1, my_s2, var_s2, var_t, Sigma_T)
    s_sample[0] = rng.multivariate_normal(my.T[0], sigma_s)
    t_sample[0] = t_guess
    a = -math.inf
    b = 0
    if(y == 1):
        a = 0
        b = math.inf
    for i in range(num_samples):
        #my_t = Sigma_T@np.array([[my_s1/var_s1 + t_sample[i-1]/var_t], [my_s2/var_s1 - t_sample[i-1]/var_t]])
        my = update_my(t_sample[i-1], my_s1, var_s1, my_s2, var_s2, var_t, Sigma_T)
        s_sample[i] = rng.multivariate_normal(my.T[0], Sigma_T)
        t_sample[i] = sp.stats.truncnorm.rvs(a, b, s_sample[i-1][0]-s_sample[i-1][1], sigma_t)

samples =[3200, 5200, 10200, 20200]

# Trying to find burnin
x = list()
for i in range(int(s_sample.__len__()/10)):
    mean = sp.stats.norm.mean(t_sample[range(i, i+100)])
    x.append(mean[0])

# Burn in
b_in = 200
s1_spred = np.mean(s_sample[200:, 0])
s1_vpred = np.std(s_sample[200:, 0])
s2_spred = np.mean(s_sample[200:, 1])
s2_vpred = np.std(s_sample[200:, 1])


x = np.linspace(0, 60, 1000)
plt.hist(s_sample[200:], bins=50, density=True)
plt.plot(x, sp.stats.norm.pdf(x, s1_spred, s1_vpred))
plt.plot(x, sp.stats.norm.pdf(x, s2_spred, s2_vpred))
plt.show()


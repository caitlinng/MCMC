#!/usr/bin/env python
# coding: utf-8

# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import scipy as sci
from scipy.stats import norm
import matplotlib as plt

'''
Markov chain that fits mean and standard deviation for a gaussian (normally distributed) sample
'''

# Produce synthetic data
mu = 3 # Mean
sd = 0.2 # SD
n = 1000 # Number of data points

data = np.random.normal(mu, sd, n)

'''
Set up MCMC (Markov Chain Monte Carlo)
'''

# Starting guess [mu, sd]
mu_guess = np.random.random()
sd_guess = np.random.random()

init_param = np.array([mu_guess, sd_guess])

# Proposal widths
w =[0.2, 0.2]

# Prior functions (prior beliefs in mu, sigma):
# That mu and sigma are between -10 and 10

# mu_prior_fun = sci.uniform(-10, 10) 
# uniform.rvs(size=n, loc = start, scale=width)
def mu_prior_fun(p):
    min_mu = -10
    max_mu = 10
    if (p > min_mu) and (p < max_mu):
        truth = 1
    else:
        truth = 0

    return truth/(max_mu-min_mu)

#sd_prior_fun = 
def sd_prior_fun(p):
    min_sd = 0.01
    max_sd = 10
    if (p > min_sd) and (p < max_sd):
        truth = 1
    else:
        truth = 0
            
    return truth/(max_sd-min_sd)

prior_funcs = [mu_prior_fun, sd_prior_fun]


# Number of iterations
n_iterates = 1000

# Define function to calculate log likelihood
def gaussian_ll(data, params): # params = [mu, sd]
    mu = params[0]
    sd = params[1]
        
    ll = 0
    for i in data:
        ll = ll + norm.logpdf(i, loc = mu, scale = sd)

    return ll

# Calculate log likelihood (ll) of initial guess
param = init_param
ll = gaussian_ll(data, param)

# Establish data store for chain
# Where first column = ll
# And second column and onwards = model parameters (in this case 2 = mu, 3 = sigma)

chain = np.zeros(shape = (n_iterates, len(param) + 1))
chain[0, 0] = ll
chain[0, 1:] = param


# In[5]:


# Run MCMC
prop_param = np.array([0, 0])

for i in range(n_iterates):
    if i % 10 == 0:
        print ('Iteration of ' + str(i) + '/' + str(n_iterates)) # Print status every ten iterations
    
    for j in range(len(param)): # Gibbs loop over number of parameters (i.e. j=1 is mu, j=2 is sd)
        # Propose a parameter value within prev. set widths
        prop_param = param.copy()

        if np.random.random() < 0.5: 
            sign = 1
        else:
            sign = -1 # randomly plus or minus width (faster than random.choice())
            
        prop_param[j] = prop_param[j] - (w[j]/2 * sign) 
        print (prop_param)
        # Calculate log likelihood of proposal
        prop_ll = gaussian_ll(data, prop_param)
        
        # Accept or reject proposal
        prior_fun = prior_funcs[j] # Grab the correct prior function (mu or sd)
        
        # Likelihood ratio
        r = np.exp(prop_ll - ll) * prior_fun(prop_param[j]) / prior_fun(param[j])
        print (np.exp(prop_ll - ll))
        print(prior_fun(prop_param[j]) / prior_fun(param[j]))
        # Is likelihood ratio less than or equal to one
        alpha = min(1, r) 
        
        # Random number between 0 to 1
        # So will have weighted chance of possibly accepting depending on how likely the new parameter is
        test = np.random.uniform(0, 1)
        # Maybe accept
        if (test < alpha): 
            ll = prop_ll
            param = prop_param.copy()
                
        # "Else" reject, though nothing to write
        
        # Store iterate
        chain[i, 0] = ll
        chain[i, 1:] = param
        
        


# In[11]:


'''
Graphs
'''
import matplotlib.pyplot as plt

# Histograms
plt.figure()
plt.hist(chain[int(n_iterates/2):,1])
plt.title('mu frequency')
print('mu = ' + str(mu))

plt.figure()
plt.hist(chain[int(n_iterates/2):,2])
plt.title('sd frequency')
print('sd = ' + str(sd))

plt.show()


# In[16]:


'''
Graphs alternative

import matplotlib.pyplot as plt

fig1, ax1 = plt.subplots()
fig1, ax2 = plt.subplots()
fig1, ax3 = plt.subplots()

ax1.set_title('Log Likelihood of Iterates')
ax2.set_title('Estimates for mu')
ax3.set_title('Estimates for sd')

plt.figure(n_iterates, chain[:,0])
plt.ylabel('ll')
plt.xlabel('iterate')

n = 0
for i in chain:
    n += 1
    ax1.plot(n, i[0])
    ax2.plot(n, i[1])
    ax3.plot(n, i[2])

ax1.set_xlabel('Iterate')
ax1.set_ylabel('Log Likelihood')

ax2.set_xlabel('Iterate')
ax2.set_ylabel('Estimate for mu')

ax3.set_xlabel('Iterate')
ax3.set_ylabel('Estimate for sd')

plt.show()
'''


# In[51]:


n = np.arange(1000+1) # Creates array of iterates

plt.figure(n, chain[:,0])
plt.ylabel('ll')
plt.xlabel('iterate')

plt.figure(n, chain[:,1])
plt.ylabel('Estimate for mu')
plt.xlabel('iterate')

plt.figure(n, chain[:,2])
plt.ylabel('Estimate for sd')
plt.xlabel('iterate')

plt.show()


# In[ ]:





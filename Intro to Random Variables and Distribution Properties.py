#!/usr/bin/env python
# coding: utf-8

# In[1]:


import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf


# #### CDF and Quantile functions

# In[2]:


# Binomial distribution
stats.binom.pmf(5, 12, 0.5)
# stats
stats.binom.mean(12, 0.5)
stats.binom.std(12, 0.5)
stats.binom.var(12, 0.5)


# In[3]:


# Normal distribution
stats.norm.cdf(0, loc = 0, scale = 1)
# Quantile function
stats.norm.ppf(0.5, loc = 0, scale = 1)
# generating random numbers
stats.norm.rvs(loc = 0, scale = 1, size = (2,3))


# What does it mean for a random variable to follow a distribution? If you *sample* from that distribution and plot a histogram of the samples, the shape of the histogram will resemble the pdf with large enough samples. Let's compare samples drawn from a standard normal distribution with samples drawn from a $t_5$ distribution.

# In[4]:


x = stats.norm.rvs(size = 10000)
y = stats.t.rvs(df = 5, size = 10000)
plt.hist(x, bins = 50, alpha = 0.5, label = 'standard normal samples', density = True)
plt.hist(y, bins = 50, alpha = 0.5, label = 't_5 samples', density = True)
plt.legend()


# Here's what a chi-square distribution looks like with increasing degrees of freedom

# In[5]:


x = np.arange(0, 10, 0.01)
for k in range(1,10):
    plt.plot(x, stats.chi2.pdf(x, df = k), label = '%d degrees of freedom' %k)
plt.ylim((0, 1))
plt.legend()


# You can create a random variable $T$ having a $t$-distribution with $n$ degrees of freedom. We need two other random variables for this:
# - $Z \sim N(0,1)$
# - $X \sim \Chi^2_n$
# Then, $T = \frac{Z}{\sqrt{X/n}}$. Let's verify this computationally.

# In[6]:


n = 5
z = stats.norm.rvs(size = 10000)
x = stats.chi2.rvs(df = n, size = 10000)
t = z / np.sqrt(x/n)
plt.hist(t, bins = 50, density=True)
plt.scatter(t, stats.t.pdf(t, df = n), s = 0.2, color='red')


# As you can see above, the histogram of our constructed variable lines up perfectly with the pdf of a $t$-distribution.

# To inspect the pdf of an $F$ distribution, we need to specify the numerator and denominator degrees of freedom.

# In[7]:


k1 = 4; k2 = 3
x = np.arange(0, 10, 0.01)
plt.plot(x, stats.f.pdf(x, dfn = k1, dfd = k2), label = 'num df: %d, denom df: %d'%(k1, k2))
plt.legend()


# #### Demonstrating LLN
# Suppose $X_1, X_2, X_3, \ldots$ are Bernoulli random variables with $P[X_i = 1] = 0.5$. Let's construct three new random variables as follows:
# $$Y_1 = \frac{X_1 + X_2 + \cdots + X_{10}}{10}$$
# $$Y_2 = \frac{X_1 + X_2 + \cdots + X_{50}}{50}$$
# $$Y_3 = \frac{X_1 + X_2 + \cdots + X_{500}}{500}$$
# What do the distributions of $Y_1$, $Y_2$ and $Y_3$ look like? For each of these, we will perform 1000 simulations, giving us 1000 possible values for each of $Y_1$, $Y_2$, and $Y_3$. Then, we'll plot a histogram of these values and compare.

# In[8]:


sample_y1 = [stats.bernoulli.rvs(p = 0.5, size = 10).mean() for _ in range(1000)]
sample_y2 = [stats.bernoulli.rvs(p = 0.5, size = 50).mean() for _ in range(1000)]
sample_y3 = [stats.bernoulli.rvs(p = 0.5, size = 500).mean() for _ in range(1000)]


# In[9]:


plt.hist(sample_y1, bins = 20, label = 'Y1 Samples')
plt.hist(sample_y2, bins = 20, label = 'Y2 Samples')
plt.hist(sample_y3, bins = 20, label = 'Y3 Samples')
plt.legend()


# Application: If $X$ follows a chi-squared distribution with $n$ degrees of freedom, then as $n$ gets large, the ratio $X/n$ starts to concentrate around 1. Here's what that looks like:

# #### Sample Statistics
# Let's generate some random data from a normal distribution to demonstrate the calculation of sample statistics.
# 
# $X_1, X_2, \ldots, X_n \sim N(0,1)$, where you can think of $n$ as the sample size.

# In[2]:


n = 100
x = stats.norm.rvs(size = n)


# Numpy has built-in methods available for arrays for calculating mean and variance.

# In[3]:


# sample mean
print(f'Sample mean = {x.mean()}')
print(f'Sample variance = {x.var()}')


# Compare these to the mean of the corresponding distribution from which the sample is drawn, which is $0$, and the variance, which is $1$. For skewness and kurtosis, we need to use functions from scipy.stats.

# In[4]:


print(f'Sample skewness = {stats.skew(x)}')
print(f'Sample kurtosis = {stats.kurtosis(x)}')


# The skewness of the distribution is $0$. The kurtosis value displayed above is the excess kurtosis, which is $0$ for the distribution. For standard deviation (volatility), we can use a Numpy method.

# In[5]:


print(f'Sample standard deviation = {x.std()}')


# To demonstrate bivariate sample statistics, we need another random sample. Let's draw this too from a standard normal distribution.
# 
# $Y_1, Y_2, \ldots, Y_n \sim N(0,1)$

# In[6]:


y = stats.norm.rvs(size = n)


# The *cov()* function in Numpy yields a covariance matrix.

# In[7]:


np.cov(x,y)


# You can extract the covariance value from the matrix:

# In[8]:


print(f'Sample covariance = {np.cov(x,y)[0,1]}')


# For the sample correlation, we use the *corrcoef()* function from Numpy.

# In[10]:


print(f'Sample correlation = {np.corrcoef(x,y)[0,1]}')


# For calculating quantiles/percentiles, we can use the *quantile()* function from Numpy.

# In[13]:


# A single quantile
np.quantile(x, 0.05)


# In[12]:


# Multiple quantiles
np.quantile(x, [0.01, 0.05])


# For the autocorrelation function, we use the *acf()* function from statsmodels. By default, the number of lags is calculated according to the length of the data.

# In[14]:


acf(x)


# In[15]:


# But you can adjust the number of lags.
acf(x, nlags = 10)


# In[18]:


# You can also include confidence intervals by providing a significance level
acf(x, nlags = 10, alpha = 0.05)


# Oftentimes it's more useful to plot the autocorrelation values. For this, we use the *plot_acf()* function from statsmodels. This also includes the 95% confidence interval.

# In[17]:


plot_acf(x)


# Not surprisingly, we don't see any large autocorrelations because the data was randomly generated. Still, notice that one or two of the autocorrelations might turn out to be significant. This is purely by chance and you shouldn't interpret it as economically meaningful.

# In[19]:


# acf of x^2
plot_acf(x**2)


# #### Variance Ratio for a Random Walk Model
# Let's draw returns, $r$ from a standard normal distribution. These are going to be independent and identically distributed.

# In[22]:


r = stats.norm.rvs(size = 1000)


# The sample obviously has a variance pretty close to $1$.

# In[23]:


r.var()


# At each time $t$, we can define the 2-period return as $r_t + r_{t-1}$. The 2-period return series is then:
# 
# $r_0 + r_1, r_2 + r_3, r_4 + r_5, \ldots$
# 
# The ratio of the variance of this series to the original variance is:

# In[24]:


np.array([r[t] + r[t-1] for t in range(1, len(r), 2)]).var() / r.var()


# Checking the variance ratio for multiple period returns from 1 to 5:

# In[28]:


for k in range(1,6):
    print(np.array([r[t-k:t+1].sum() for t in range(1, len(r), k+1)]).var() / ((k+1) * r.var()))


# In[10]:


for n in range(10, 200, 10):
    x = stats.chi2.rvs(df = n, size = 10000)
    plt.hist(x/n, bins = 50, label = 'n = %d'%n, density = True)
plt.xlim((0,5))
plt.legend()


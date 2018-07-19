
# coding: utf-8

# In[22]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import figure
from scipy.stats import lognorm, halfgennorm, powerlaw
import numpy as np
# from fitter import Fitter


# In[23]:


original_df = pd.read_csv('data/original/fm_interactions.csv', delimiter='\t')


# In[24]:


original_df.head()


# In[25]:


"""
There are five classes of interaction type per user.  These classes are:

0 = XING showed this item to a user (= impression)

1 = the user clicked on the item

2 = the user bookmarked the item on XING

3 = the user clicked on the reply button or application form button that is shown on some job postings

4 = the user deleted a recommendation from his/her list of recommendation (clicking on "x")
    which has the effect that the recommendation will no longer been shown to the user and that a new 
    recommendation item will be loaded and displayed to the user
    
5 = a recruiter from the items company showed interest into the user. (e.g. clicked on the profile)
"""
print(sorted(original_df['interaction_type'].unique()))


# In[26]:


"""
There are 1,096,350 unique users in the dataset
"""
print(len(original_df.user_id.unique()))


# In[27]:


print(original_df.shape)


# In[28]:


users_df = pd.read_csv('data/original/users.csv', delimiter='\t')


# In[29]:


users_df.head()


# In[30]:


users_df.shape


# In[31]:


"""
Filter out interactions outside of the categories 1, 2 and 3.  These three categories are all interactions
that suggest user optimism.

Once the categories are filtered, group the interactions by user_id to get the total count of interactions
for each user.
"""
interaction_df = original_df[original_df['interaction_type'].isin([1, 2, 3])]
user_group = interaction_df.groupby(['user_id'])
user_interaction_count = user_group['item_id'].count()


# In[32]:


"""
Plot the distribution.

We see in the plot below that the vast majority of users have a number of interactions between 1 and 100,
with a small number of outliers.
"""
fig = plt.figure(figsize=(10, 10))
graph = sns.distplot(user_interaction_count)
graph.set_title('User Interaction Count Histogram')
graph.set(ylabel='Percent', xlabel='Interaction Count')
plt.show()


# In[33]:


"""
After removing the outlier interactions, we see that our users usually have interactions between 0 and 40 times.
"""
fig = plt.figure(figsize=(10, 10))
clipped_interactions = user_interaction_count.clip(0, 51)
graph = sns.distplot(clipped_interactions)
graph.set_title('User Interaction Count Histogram (clipped at 50)')
graph.set(ylabel='Percent', xlabel='Interaction Count')
plt.show()


# In[10]:


""" Using the 'Fitter' library: https://pypi.python.org/pypi/fitter - we can get a ranking of what distributions fit
    the dataset well.  One should note that empty bins are taken into account for the mean error, so it is beneficial
    to be mindful that your bins should follow a continuous trend.
"""
f = Fitter(
    clipped_interactions,
    distributions=['betaprime', 'halfgennorm', 'kappa3', 'bradford', 'lognorm', 'powerlaw'],
    bins=50,
    xmin=1,
    xmax=50)
f.fit()
f.summary()
plt.show()


# In[38]:


"""
The empty bins threw off the algorithm quite a bit.  If we reduce the number of bins to 30, we see that our
distribution continuously decreases, suggesting our fit will be more accurate.
"""

clipped_interactions = user_interaction_count.clip(1, 31)
fig = plt.figure(figsize=(10, 10))
graph = sns.distplot(clipped_interactions, bins = 30)
graph.set_title('User Interaction Count Histogram (clipped at 30)')
graph.set(ylabel='Percent', xlabel='Interaction Count')
plt.show()


# In[39]:


fig = plt.figure(figsize=(10, 10))
f = Fitter(
    clipped_interactions,
    # distributions=['betaprime', 'halfgennorm', 'kappa3', 'bradford', 'lognorm', 'powerlaw'],
    bins=30,
    xmin=1,
    xmax=31,
    timeout=120,
)
f.fit()
f.summary()
plt.show()


# In[35]:


shape, loc, scale = f.fitted_param['halfgennorm']
x = np.linspace(halfgennorm.ppf(0.01, *f.fitted_param['halfgennorm']),
                halfgennorm.ppf(0.99, *f.fitted_param['halfgennorm']), 100)
plt.plot(x - loc, halfgennorm.pdf(x, *f.fitted_param['halfgennorm']),
       'r-', lw=1, alpha=0.6, label='lognorm pdf')
plt.show()
print(halfgennorm.pdf(1.5, *f.fitted_param['halfgennorm']))
print(shape, loc, scale)


# In[40]:


"""
The 'fit' function does not appear to work well with the lognormal distribution.  After trial and error, it appears
that a value close to 1.46 fits our distribution quite well.
"""

fig = plt.figure(figsize=(10, 10))
sns.distplot(clipped_interactions, bins=30)
sns.set_title('Lognorm PDF')
sns.set(xlabel='Interaction Count', ylabel='Percent')
s = 1.46
x = np.linspace(lognorm.ppf(0.01, s),
                lognorm.ppf(0.99, s), 100)
plt.plot(x, lognorm.pdf(x, s),
       'r-', lw=1, alpha=0.6, label='lognorm pdf')
plt.show()


# In[12]:


delta = .001
lognorm_dist = lognorm(s)
upper_lognorm_dist = lognorm(s + delta)
lower_lognorm_dist = lognorm(s - delta)


# In[13]:


upper_prob = upper_lognorm_dist.pdf(1)
lower_prob = lower_lognorm_dist.pdf(1)
print(upper_lognorm_dist.pdf(1))
print(lower_lognorm_dist.pdf(1))
print(lognorm_dist.pdf(1))
print((upper_prob + lower_prob) / 2)


# In[19]:


"""
If we adjust the 's' parameter - or the variance in the distribution, we see that we can create two sets of user 
distributions - one in which users are more likely to click more, another in which users are more likely to click
less.  Still, from the results above, we see that the two distributions do not average out to the old distribution.
Our goal is for the two distributions to SUM to the original distribution.  The math for a lognormal distribution
is a little tricky, suggesting a power law distributuion would be best suited for this challenge.
"""

fig = plt.figure(figsize=(10, 10))

sns.distplot(clipped_interactions, bins=30)

delta = .5
s = 1.46

x = np.linspace(lognorm.ppf(0.01, s),
                lognorm.ppf(0.99, s), 100)
plt.plot(x, lognorm.pdf(x, s + delta),
       'r-', lw=1, alpha=0.6, label='upper lognorm pdf')

plt.plot(x, lognorm.pdf(x, s - delta),
       'r-', lw=1, alpha=0.6, label='lower lognorm pdf')


plt.show()


# In[21]:


fig = plt.figure(figsize=(10, 10))
power_fit = Fitter(
    clipped_interactions,
    distributions=['powerlaw', 'exponpow', 'powerlognorm', 'powernorm'],
    bins=30,
    xmin=1,
    xmax=30,
    timeout=120,
)
power_fit.fit()
power_fit.summary()
plt.show()


# In[24]:


"""
We see that a is close to .2, the axis is adjusted by one (which makes sense because we have no 0 values), and scale
is adjusted by a factor of 30
"""

powerlaw_params = power_fit.fitted_param['powerlaw']
print(powerlaw_params)
pl_a, pl_loc, pl_scale = powerlaw_params


# In[33]:


# Lets make sure we can plot this correctoy
sns.distplot(clipped_interactions, bins=30)
x = np.linspace(1, 30, 100)
plt.plot(x, powerlaw.pdf(x, *powerlaw_params),
       'r-', lw=1, alpha=0.6, label='upper lognorm pdf')

plt.show()


# In[34]:


"""
This graph shows it is fine to generalize with 0.2 for the exponent and 30 for the scale parameter.  We need to use
.99999 instead of 1 for loc parameter because 1 will result in a divide by 0 error.
"""
sns.distplot(clipped_interactions, bins=30)
x = np.linspace(1, 30, 100)
plt.plot(x, powerlaw.pdf(x, .2, .999999999, 30),
       'r-', lw=1, alpha=0.6, label='upper lognorm pdf')

plt.show()


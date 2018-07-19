
# coding: utf-8

# In[34]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import figure
from scipy.stats import powerlaw, halfgennorm
import numpy as np
from math import log
from random import random


# In[35]:


"""
This is the state of the data from where we left off in 'distribution_analysis'

The analysis showed that clipping the distribution between 1 and 30 clicks allowed the best fit using the
powerlaw distribution.
"""
original_df = pd.read_csv('data/original/fm_interactions.csv', delimiter='\t')
interaction_df = original_df[original_df['interaction_type'].isin([1, 2, 3])]
user_group = interaction_df.groupby(['user_id'])
user_interaction_count = user_group['item_id'].count()
clipped_interactions = user_interaction_count.clip(1, 31)


# In[36]:


"""
These are the fitted params for the fitted halfgennorm distribution
"""
fitted_params = (0.325, .9, .2, )


# In[37]:


"""
For starters we'll plot the halfgennorm probability distribution function in its original form...
"""
x = np.linspace(1, 30, 1000)
plt.plot(x, halfgennorm.pdf(x, *fitted_params),
       'r-', lw=1, alpha=0.6, label='powerlow pdf')
sns.distplot(clipped_interactions, bins=30)
plt.xlim(0, 30)
plt.ylim(0, .5)
plt.show()
print(halfgennorm.pdf(1, *fitted_params))


# In[38]:


powerlaw_params = (0.45, 0.9, 30)


# In[39]:


x = np.linspace(1, 30, 1000)
plt.plot(x, powerlaw.pdf(x, *powerlaw_params),
       'r-', lw=1, alpha=0.6, label='powerlow pdf')
sns.distplot(clipped_interactions, bins=30)
plt.xlim(0, 30)
plt.ylim(0, .5)
plt.show()
print(powerlaw.pdf(1, *powerlaw_params))


# In[44]:


"""
Now that we have a model of our distribution, we want to create a piecewise function for two synthetic distributions.
The sum of the synthetic distributions will equal the original distribution.


C(x) = Original Distribution
LAMBDA = Cutoff Variable

K = C(LAMBDA)

A = {
        K                IF x <= LAMBDA
        C(x)             IF x > LAMBDA
}

B = {
        C(x) - K         IF x <= LAMBDA
        0                IF x > LAMBDA
}
"""

LAMBDA = 14
dist = halfgennorm
k = dist.pdf(LAMBDA, *fitted_params)

up_to_lambda = np.linspace(1, LAMBDA, LAMBDA)
passed_lambda = np.linspace(LAMBDA + 1, 30, 30 - LAMBDA)

B = halfgennorm.pdf(up_to_lambda, *fitted_params) - k
B = np.append(B, np.zeros(30 - LAMBDA))

A = halfgennorm.pdf(passed_lambda, *fitted_params)
A = np.append(np.ones(LAMBDA) * k, A)

fixed_a = sum(A)
fixed_b = sum(B)


# In[45]:


fig = plt.figure(figsize=(8, 8))
x = np.linspace(1, 30, 30)
plt.plot(x, A, 'r-', lw=1, alpha=0.6, label='P(A)')
plt.plot(x, B, 'b-', lw=1, alpha=0.6, label='P(B)')
plt.title('Original PDF of groups A and B with lambda at 14')
plt.xlabel('# of user interactions / lambda')
plt.ylabel('Probability')
fig.legend()
plt.show()


# In[46]:


fig = plt.figure(figsize=(8, 8))
x = np.linspace(1, 30, 30)
plt.plot(np.log(x), np.log(A), 'r-', lw=1, alpha=0.6, label='A pdf')
plt.plot(np.log(x), np.log(B), 'b-', lw=1, alpha=0.6, label='B pdf')
plt.title('Adjusted PDF of groups A and B at lambda + 2 (16)')
plt.xlabel('Log # of user interactions / lambda')
plt.ylabel('Log Probability')
fig.legend()
plt.show()


# In[42]:


LAMBDA = 16
dist = halfgennorm
k = dist.pdf(LAMBDA, *fitted_params)

up_to_lambda = np.linspace(1, LAMBDA, LAMBDA)
passed_lambda = np.linspace(LAMBDA + 1, 30, 30 - LAMBDA)

B = dist.pdf(up_to_lambda, *fitted_params) - k
B = np.append(B, np.zeros(30 - LAMBDA))

A = halfgennorm.pdf(passed_lambda, *fitted_params)
A = np.append(np.ones(LAMBDA) * k, A)

new_a = sum(A)

a_adj = fixed_a - new_a

N = LAMBDA - 1
x_int = (a_adj * 2) / (N * (N + 1))

A_line = [k + x_int * (N - x) for x in range(0, LAMBDA)]

A = halfgennorm.pdf(passed_lambda, *fitted_params)
A = np.append(A_line, A)

B = halfgennorm.pdf(up_to_lambda, *fitted_params) - A_line
B = np.append(B, np.zeros(30 - LAMBDA))


# In[43]:


fig = plt.figure(figsize=(8, 8))
x = np.linspace(1, 30, 30)
plt.plot(x, A, 'r-', lw=1, alpha=0.6, label='A pdf')
plt.plot(x, B, 'b-', lw=1, alpha=0.6, label='B pdf')
plt.title('Adjusted PDF of groups A and B at lambda + 2 (16)')
plt.xlabel('# of user interactions / lambda')
plt.ylabel('Probability')
fig.legend()
plt.show()


# In[11]:


"""
Lets create a dataframe that pairs users with their corresponding interaction count
"""
user_df = clipped_interactions.to_frame()
user_df.columns = ['interaction_count']
user_df['user_id'] = user_df.index
user_df.head()


# In[12]:


interaction_count_df = user_df.groupby(['interaction_count']).count()
interaction_count_df.columns = ['num_users']
interaction_count_df['interaction_count'] = interaction_count_df.index
interaction_count_df.head()


# In[19]:


"""
Now we want to search through different values of lambda to see what our original ratio is.  We can set the initial
lambda to be a ratio that makes sense.  In our case, it is likely the ratio between A and B that makes the most sense
is around 50%.  In that case, the lambda value that we want to start with is 7.
"""
synthetic_prob_a_dict = {}
dist = halfgennorm
for lambda_val in range(1, 32):
    k = dist.pdf(lambda_val, *fitted_params)

    up_to_lambda = np.linspace(1, lambda_val, lambda_val)
    passed_lambda = np.linspace(lambda_val + 1, 31, 31 - lambda_val)

    B = halfgennorm.pdf(up_to_lambda, *fitted_params) - k
    B = np.append(B, np.zeros(31 - lambda_val))

    A = halfgennorm.pdf(passed_lambda, *fitted_params)
    A = np.append(np.ones(lambda_val) * k, A)
    prob_a = sum(A)/(sum(A) + sum(B))
    synthetic_prob_a_dict[lambda_val] = prob_a


# In[20]:


"""
What we really care about is the ratio that we can expect to get on the actual dataset.  Using the expected probability
at each ratio, we can multiply it by the true user count at each lambda to get the expected number of users that
fall into category A.  From there we can get the overall probability a user will fall into group A if we use
our synthetic distribution to derive the group each user belongs to.  We see that here too the lambda with an even
ratio is 7, and that generally our probabilities line up, which is good.  It means that our synthetic probability does
a good job of simulating our user data.
"""
real_prob_a_dict = {}
dist = halfgennorm
for lambda_val in range(1, 32):
    k = dist.pdf(lambda_val, *fitted_params)

    up_to_lambda = np.linspace(1, lambda_val, lambda_val)
    passed_lambda = np.linspace(lambda_val + 1, 31, 31 - lambda_val)

    B = halfgennorm.pdf(up_to_lambda, *fitted_params) - k
    B = np.append(B, np.zeros(31 - lambda_val))

    A = halfgennorm.pdf(passed_lambda, *fitted_params)
    A = np.append(np.ones(lambda_val) * k, A)
    
    prob_A = A / (A + B)

    lambda_mapping = {'interaction_count': range(1, 32), 'prob_a': [prob_A[idx] for idx in range(31)]}

    prob_A_df = pd.DataFrame.from_dict(lambda_mapping)
    
    # we assign the item to group A if the probability is greater than 0.5 for now. We will assign A randomly
    # later, when we have decided the ideal lambda
    prob_count_df = prob_A_df.merge(interaction_count_df, on=['interaction_count'])
    prob_count_df['expectation_a'] = prob_count_df['prob_a'] * prob_count_df['num_users']
    sum_df = prob_count_df.sum()
    a_prob = sum_df['expectation_a'] / sum_df['num_users']
    real_prob_a_dict[lambda_val] = a_prob


# In[32]:


fig = plt.figure(figsize=(8, 8))
plt.plot(real_prob_a_dict.keys(), real_prob_a_dict.values(), label='Real')
plt.plot(synthetic_prob_a_dict.keys(), synthetic_prob_a_dict.values(), label='Synthetic')
plt.title('Expectation of Group A vs. lambda')
plt.xlabel('lambda')
plt.ylabel('Expectation')
plt.legend()
plt.plot()
plt.show()


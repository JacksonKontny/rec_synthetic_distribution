
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
from scipy.stats import halfgennorm
import numpy as np
from math import log
from random import random


# In[2]:


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
user_df = clipped_interactions.to_frame()
user_df.columns = ['interaction_count']
user_df.head()


# In[3]:

"""
Lets create a dataframe that pairs users with their corresponding interaction count
"""
user_df = clipped_interactions.to_frame()
user_df.columns = ['interaction_count']
user_df['user_id'] = user_df.index
user_df.head()

# In[4]:

"""
These are the fitted params for the fitted halfgennorm distribution
"""
fitted_params = (0.325, .9, .2, )

# In[5]:


"""
This is the 'original lambda'.  See 'piecewise_distribution' for mor about this.  We use a lambda of 7, which
corresponds to an overall probability the user is assigned to group A of around 49%.  We can see that after running
our similation the probability of A ends up being around 49%, which is good.
"""

LAMBDA = 7
dist = halfgennorm
k = dist.pdf(LAMBDA, *fitted_params)

up_to_lambda = np.linspace(1, LAMBDA, LAMBDA)
passed_lambda = np.linspace(LAMBDA + 1, 31, 31 - LAMBDA)

B = halfgennorm.pdf(up_to_lambda, *fitted_params) - k
B = np.append(B, np.zeros(31 - LAMBDA))

A = halfgennorm.pdf(passed_lambda, *fitted_params)
A = np.append(np.ones(LAMBDA) * k, A)

prob_A = A / (A + B)

lambda_mapping = {'interaction_count': range(1, 32), 'prob_a': [prob_A[idx] for idx in range(31)]}

prob_A_df = pd.DataFrame.from_dict(lambda_mapping)
user_prob_df = user_df.merge(prob_A_df, on='interaction_count')

user_prob_df['is_a'] = user_prob_df['prob_a'].map(lambda x: x >= random())
print(sum(user_prob_df['is_a']) / len(user_prob_df['is_a']))


# In[9]:


user_prob_df.to_csv('data/synthetic/piecewise_lambda_7_to_7.csv', index=False)


# In[54]:


"""
We now want to generate some datasets with different values for lambda.
"""
LAMBDA = 7
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


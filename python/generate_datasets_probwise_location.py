
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
from scipy.stats import powerlaw
import numpy as np
from math import log
from random import random

# In[]
df = pd.read_csv('data/original/users.csv', delimiter='\t')
# In[]
df.columns = [
    'id',
    'jobroles',
    'career_level',
    'discipline_id',
    'industry_id',
    'country',
    'region',
    'experience_n_entries_class',
    'experience_years_experience',
    'experience_years_in_current',
    'edu_degree',
    'edu_fieldofstudies',
    'wtcj',
    'premium']
# In[]
career_count = df.groupby(['country', 'region']).count()['career_level']
out_df = career_count / sum(career_count)
out_df.to_csv('region_membership.csv')
# In[2]:
"""
It appears career count for region 7 makes up around 6% of the dataset.
We will use this region so reduce the number of users in the dataset.
"""
df = df[df['region'] == 7]
df = df[df['country'] == 'de']
df = df[df['career_level'] == 0]
filtered_user_ids = df[['id']]
filtered_user_ids.head()
# In[]

"""
This is the state of the data from where we left off in 'distribution_analysis'

The analysis showed that clipping the distribution between 1 and 30 clicks allowed the best fit using the
powerlaw distribution.
"""
original_df = pd.read_csv('../data/original/fm_interactions.csv', delimiter='\t')
interaction_df = original_df[original_df['interaction_type'].isin([1, 2, 3])]
user_group = interaction_df.groupby(['user_id'])
user_interaction_count = user_group['item_id'].count()
clipped_interactions = user_interaction_count.clip(1, 31)
user_df = clipped_interactions.to_frame()
user_df.columns = ['interaction_count']
user_df['user_id'] = user_df.index

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
These are the fitted params for the fitted powerlaw distribution
"""
fitted_params = (0.45, 0.9, 30)

# In[5]:


"""
This is the 'original lambda'.  See 'probability_distribution' for more about this.  We use an initial, 
probability of 0.58 which corresponds to an overall probability the user is assigned to group A of around 50%. 
We can see that after running our similation the probability of A ends up being around 49%, which is good.
"""

dist = powerlaw
initial_a_to_b_ratio = .58

buckets = 31

x = np.linspace(1, buckets, buckets)

C = dist.pdf(x, *fitted_params)

B = np.multiply((buckets - x + 1) * initial_a_to_b_ratio / buckets, C)

A = C - B

prob_A = A / (A + B)

lambda_mapping = {'interaction_count': range(1, 31), 'prob_a': [prob_A[idx] for idx in range(30)]}

prob_A_df = pd.DataFrame.from_dict(lambda_mapping)
user_prob_df = user_df.merge(prob_A_df, on='interaction_count')

user_prob_df['is_a'] = user_prob_df['prob_a'].map(lambda x: x >= random())
print(sum(user_prob_df['is_a']) / len(user_prob_df['is_a']))

# In[]
user_prob_df = filtered_user_ids.merge(user_prob_df, left_on='id', right_on='user_id', how='inner')

# In[10]:
librec_membership_df = user_prob_df[['user_id', 'is_a']]
librec_boolean_map = {True: 1, False: -1}
librec_membership_df['is_a'] = librec_membership_df['is_a'].apply(lambda x: librec_boolean_map[x])
librec_membership_df.head()

# In[]
librec_membership_df['is_a'].unique()

# In[9]:
librec_membership_df.to_csv('../FAW-RS-test/data/membership/xing_location/membership_xing_demographic.txt', index=False)
user_prob_df.to_csv('data/synthetic/probability_location.csv', index=False)

# In[]
all_ratings = pd.read_csv('data/all_ratings.csv')
# In[]
all_ratings.shape


# In[]
merged_df = filtered_user_ids.merge(all_ratings, left_on='id', right_on='user_id', how='inner')

# In[]
merged_df = merged_df[['user_id', 'item_id', '0']]
merged_df.to_csv('FAW-RS-test/data/xing_58_de_7/ratings.txt', header=False, index=False, sep='\t')

# coding: utf-8

# In[3]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import figure
from scipy.stats import powerlaw, halfgennorm
import numpy as np
from math import log
from pprint import pprint as pp


# In[4]:


"""
This is the state of the data from where we left off in 'distribution_analysis'

The analysis showed that clipping the distribution between 1 and 30 clicks allowed the best fit using the
powerlaw distribution.
"""
original_df = pd.read_csv('data/original/fm_interactions.csv', delimiter='\t')
interaction_df = original_df[original_df['interaction_type'].isin([1, 2, 3])]
user_group = interaction_df.groupby(['user_id'])
user_interaction_count = user_group['item_id'].count()
clipped_interactions = user_interaction_count.clip(1, 30)
# In[7]:
powerlaw_params = (0.45, 0.9, 30)

# In[8]:
x = np.linspace(1, 30, 1000)
plt.plot(x, powerlaw.pdf(x, *powerlaw_params),
       'r-', lw=1, alpha=0.6, label='powerlow pdf')
sns.distplot(clipped_interactions, bins=30)
plt.xlim(0, 30)
plt.ylim(0, .5)
plt.show()

# In[9]:
dist = powerlaw
fitted_params = powerlaw_params
initial_a_to_b_ratio = .55

buckets = 30

x = np.linspace(1, buckets, buckets)
C = dist.pdf(x, *fitted_params)
B = np.multiply((buckets - x + 1) * initial_a_to_b_ratio / buckets, C)
A = C - B

# In[10]:

fig = plt.figure(figsize=(8, 8))
x = np.linspace(1, buckets, buckets)
plt.plot(x, A, 'r-', lw=1, alpha=0.6, label='A pdf')
plt.plot(x, B, 'b-', lw=1, alpha=0.6, label='B pdf')
plt.plot(x, C, 'g-', lw=1, alpha=0.6, label='C pdf')
plt.title(
    'Ratio distribution scheme of groups A and B justaposed against original '
    'distribution C'
)
plt.xlabel('# of user interactions')
plt.ylabel('Probability')
fig.legend()
plt.show()

# In[14]:

user_df = clipped_interactions.to_frame()
user_df.columns = ['interaction_count']
user_df['user_id'] = user_df.index
interaction_count_df = user_df.groupby(['interaction_count']).count()
interaction_count_df.columns = ['num_users']
interaction_count_df['interaction_count'] = interaction_count_df.index

# In[19]:

real_prob_a_dict = {}
prob_data_dict = {}
synthetic_prob_a_dict = {}
buckets = 30

x = np.linspace(1, buckets, buckets) # x = [1, ..., 30]

def get_prob_dfs(bucket_count, ratio, fitted_params, desired_prob_a=None, actual_prob_a=None):
    bucket_indices = np.linspace(1, bucket_count, bucket_count) # x = [1, ..., 30]
    C = dist.pdf(bucket_indices, *fitted_params) # C is the original distribution
    C = C/sum(C)

    # Example:
    # buckets = 30
    # ratio = .5
    # x = [1, ..., buckets]
    # First value in B will be: ((30 -  1 + 1) * 0.5) / 30, which is (C * 0.5) * (30 / 30)
    # 2nd  value in B will be:  ((30 -  2 + 1) * 0.5) / 30, which is (C * 0.5) * (29 / 30)
    # ....                                   
    # Last value in B will be:  ((30 - 30 + 1) * 0.5) / 30, which is (C * 0.5) * ( 1 / 30)
    B = np.multiply(
        (bucket_count - bucket_indices + 1) * ratio / bucket_count,
        C
    )
    A = C - B
    return A, B, C

def get_interaction_count_prob_df(A, B, bucket_count):
    """ Returns a dataframe with the probability a user will be in Group A

        Ex:

            interaction_count	prob_a
        0	1	                0.020000
        1	2	                0.052667
        2   3                   1.0
    """
    prob_A = A / (A + B)
    lambda_mapping = {
        'interaction_count': range(1, bucket_count + 1),
        'prob_a': [prob_A[idx] for idx in range(bucket_count)]
    }
    prob_A_df = pd.DataFrame.from_dict(lambda_mapping)
    prob_count_df = prob_A_df.merge(interaction_count_df, on=['interaction_count'])
    return prob_count_df

# In[19]:

def get_rating_counts(prob_count_df):
    # we assign the item to group A if the probability is greater than 0.5 for now.
    # We will assign A randomly later, when we have decided the ideal lambda
    prob_count_df['a_user_count'] = prob_count_df['num_users'] * prob_count_df['prob_a']
    prob_count_df['b_user_count'] = (
        prob_count_df['num_users'] * (1 - prob_count_df['prob_a'])
    )
    prob_count_df['a_rating_count'] = (
        prob_count_df['a_user_count'] * prob_count_df['interaction_count']
    )
    prob_count_df['b_rating_count'] = (
        prob_count_df['b_user_count'] * prob_count_df['interaction_count']
    )
    return prob_count_df

def get_overall_probability(prob_count_df):
    sum_df = prob_count_df.sum()
    a_prob = sum_df['a_user_count'] / sum_df['num_users']
    return a_prob

def get_avg_rating_ratio(prob_count_df):
    sum_df = prob_count_df.sum()
    a_ratings_avg = sum_df['a_rating_count'] / sum_df['a_user_count']
    b_ratings_avg = sum_df['b_rating_count'] / sum_df['b_user_count']
    return a_ratings_avg / b_ratings_avg

# In[22]:
adj_prob_data_dict = {}
original_prob_data_dict = {}
desired_prob_a = 0.5
for ratio in np.arange(0.02, 1.00, .02):
    A, B, C = get_prob_dfs(buckets, ratio, fitted_params)
    
    synthetic_prob_a_dict[ratio] = sum(A) / (sum(A) + sum(B))

    prob_A_df = get_interaction_count_prob_df(A, B, buckets)
    prob_count_df = get_rating_counts(prob_A_df)
    actual_prob_a = get_overall_probability(prob_count_df)


    adjustment_ratio = desired_prob_a / actual_prob_a
    original_avg_rating_ratio = get_avg_rating_ratio(prob_count_df)
    original_prob_data_dict[ratio]  = {
        'a_prob': actual_prob_a,
        'avg_rating_ratio': original_avg_rating_ratio
    }
    adj_A = A * adjustment_ratio
    adj_B = C - adj_A
    adj_prob_A_df = get_interaction_count_prob_df(adj_A, adj_B, buckets)
    adj_count_df = get_rating_counts(adj_prob_A_df)
    actual_prob_a = get_overall_probability(prob_count_df)
    if (adj_count_df['prob_a'] >= 1).any():
        print(ratio, 'failed')
        continue

    adj_prob_data_dict[ratio]  = {
        'a_prob': get_overall_probability(adj_count_df), 
        'avg_rating_ratio': get_avg_rating_ratio(adj_count_df),
        'original_avg_rating_ratio': original_avg_rating_ratio,
        'adjustment': adjustment_ratio,
        'df': adj_prob_A_df,
    }

# In[22]:
adj_prob_data_dict
# In[22]:
for idx, values in adj_prob_data_dict.items():
    print(idx, '\t', values['avg_rating_ratio'], values['original_avg_rating_ratio'])

pd.DataFrame(adj_prob_data_dict).T.to_csv('rating_ratio_data.csv')


# In[22]:

# In[22]:
fig = plt.figure(figsize=(8, 8))
plt.plot(real_prob_a_dict.keys(), real_prob_a_dict.values(), label='Real')
plt.plot(
    synthetic_prob_a_dict.keys(),
    synthetic_prob_a_dict.values(),
    label='Synthetic'
)
plt.title('Expectation of Group A vs. k')
plt.xlabel('k')
plt.ylabel('Expectation')
plt.legend()
plt.plot()
plt.show()

# In[23]:
x = adj_prob_data_dict.keys()
y = [value['avg_rating_ratio'] for value in adj_prob_data_dict.values()]
fig = plt.figure(figsize=(8, 8))
plt.plot(x, y, label='Adjusted Rating Ratio')

x = original_prob_data_dict.keys()
y = [value['avg_rating_ratio'] for value in original_prob_data_dict.values()]
plt.plot(x, y, label='Original Rating Ratio')

plt.title('Compared Rating Ratio')
plt.xlabel('K')
plt.ylabel('Rating Ratio')
plt.legend()
plt.plot()
plt.show()

# In[23]:
len(sorted(user_df['interaction_count'].unique()))
sorted(ten_percent_df['interaction_count'].unique())

# In[23]:
# 0.12000000000000001           1.10
# 0.38                          1.50
# 0.62                          2.67
# 0.8400000000000001            55.44
# In[23]:

ten_percent_df = adj_prob_data_dict[0.12000000000000001]['df']
fifty_percent_df = adj_prob_data_dict[0.38]['df']
max_df = adj_prob_data_dict[0.62]['df']
# In[23]:
average_profile_size = sum(ten_percent_df['interaction_count'] * ten_percent_df['num_users']) / sum(ten_percent_df['num_users'])
# In[23]:
print(1.1 * average_profile_size * 2 / 2.1)
print(1 * average_profile_size * 2 / 2.1)
print('\n')
print(1.5 * average_profile_size * 2 / 2.5)
print(1 * average_profile_size * 2 / 2.5)
print('\n')
print(2.6 * average_profile_size * 2 / 3.6)
print(1 * average_profile_size * 2 / 3.6)
print('\n')

# In[23]:
def create_prob_dataset(df, csv_name):
    print(user_df.shape)
    user_prob_df = user_df.merge(df, on='interaction_count')
    print(user_prob_df.shape)
    user_prob_df['is_a'] = user_prob_df['prob_a'].map(lambda x: x >= random())
    user_prob_df = user_prob_df[['user_id', 'is_a']]
    librec_boolean_map = {True: -1, False: 1}
    user_prob_df['is_a'] = user_prob_df['is_a'].apply(lambda x: librec_boolean_map[x])
    user_prob_df.to_csv('data/{}.csv'.format(csv_name), index=False, header=False, sep=',')

# In[24]:
create_prob_dataset(ten_percent_df, 'ten_percent_membership')
create_prob_dataset(fifty_percent_df, 'fifty_percent_membership')
create_prob_dataset(max_df, 'max_membership')
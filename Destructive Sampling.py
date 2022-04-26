#!/usr/bin/env python
# coding: utf-8

# # Reliability Inspection

# In[1]:


import numpy as np
import math
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from fractions import Fraction
from decimal import Decimal


# In[2]:


colors = sns.color_palette("colorblind")


# In[3]:


# pip install dataframe_image


# Buy n lightbulbs and wait until m fail

# In[4]:


def m_from_n(m, n, p):
    lightbulbs = [1] * n
    count = 0
    episodes = 0
    while(count < m):
        episodes += 1
        for i in range(n):
            if(count >= m):
                return episodes
            if random.random() <= p:
                lightbulbs[i] = 0
                count += 1
    return episodes


# In[5]:


m_from_n(10, 100, 0.1)


# Buy lightbulbs one at a time and wait until m lightbulbs fail

# In[6]:


def m_sample(m, p):
    count = 0
    episodes = 0
    while(count < m):
        episodes += 1
        if random.random() <= p:
            count += 1
    return episodes


# In[7]:


m_sample(10, 0.1)


# In[8]:


def trial(n,p):
#     print("trial")
    x = []
    test_1 = []
    test_2 = []
    for i in range(100):
    #         n = n*(i+1)
        m = math.ceil(n*(i/100))
#         print("Trial with m=",m, ", n=",n)
        x.append(m)
        test_1.append(m_from_n(m, n, p))
        test_2.append(m_sample(m, p))
    if(False):
        print("test 1",test_1)
        print("test 2",test_2)
        print("x",x)
    return [x, test_1, test_2]


# In[9]:


def run_k_trials(m, n, k, p):
    sums = [np.zeros(100)] * 3
    avg = []
    for i in range(k):
        data = trial(n, p)
        sums[0] = data[0]
        sums[1] = np.add(sums[1], data[1])
#         sums[2] = np.add(sums[2], data[2])
    avg.append(data[0])
    avg.append(np.true_divide(sums[1], k))
    avg.append(np.true_divide(sums[2], k))
    return avg
        


# In[10]:


def create_plot(data):
    x = data[0]
    y1 = data[1]
    y2 = data[2]
    
    plt.plot(x, y1, label = "Test 1")
    plt.plot(x, y2, label = "Test 2")
    
    plt.xlabel('n')
    plt.ylabel('# of Episodes')
    plt.title('M-sample vs M:N-sample')
    plt.legend()
    plt.show()


# In[11]:


data = run_k_trials(10, 10, 50, 0.01)
# create_plot(data)


# ## Distribution with Time

# In[12]:


def calc_stats(times):
    arr = np.array(times)
    avg_time = round(np.mean(arr), 2)
    std_dev = round(np.std(arr), 2)
    maximum = round(np.amax(arr), 2)
    minimum = round(np.amin(arr), 2)
    return {"avg_time":avg_time,"std_dev":std_dev,"maximum":maximum,"minimum":minimum}


# In[13]:


def m_from_n_with_time(m, n, p):
    time_between_failures = [0]
    avg_time = 0
    last_failure_episode = 0
    episode = 0
    failures = 0
    while(failures < m):
        episode += 1
        for i in range(n):
            if(failures >= m):
                return {'num_episodes':episode,'stats':calc_stats(time_between_failures),'time':time_between_failures}
            if (random.random() <= p):
                failures += 1
                time_between_failures.append(episode - last_failure_episode)
                last_failure_episode = episode
    return {'num_episodes':episode,'stats':calc_stats(time_between_failures),'time':time_between_failures}


# In[14]:


m_from_n_with_time(10, 20, 0.01)


# In[15]:


def m_sample_with_time(m, p):
    time_between_failures = [0]
    last_failure_episode = 0
    failures = 0
    episode = 0
    while(failures < m):
        episode += 1
        if random.random() <= p:
            failures += 1
            time_between_failures.append(episode - last_failure_episode)
            last_failure_episode = episode
    return {'num_episodes':episode,'stats':calc_stats(time_between_failures),'time':time_between_failures}


# In[16]:


m_sample_with_time(10, 0.01)


# In[17]:


def trial_with_time(id_num, m, n, p):
    return {'id':id_num, 'm':m, 'n':n, 'p':p,'test_1':m_from_n_with_time(m, n, p), 'test_2':m_sample_with_time(m, p)}


# In[18]:


print(trial_with_time(0, 10, 20, 0.01))


# In[19]:


data = trial_with_time(0, 10, 20, 0.01)

x = np.arange(len(data['test_1']['time']))
y1 = np.array(data['test_1']['time'])
plt.scatter(x, y1, color=colors[0], label='Test 1')

x = np.arange(len(data['test_2']['time']))
y2 = np.array(data['test_2']['time'])
plt.scatter(x, y2, color=colors[1], label='Test 2')

plt.xlabel('ith failure')
plt.ylabel('# of Episodes')
plt.title('M-sample vs M:N-sample')
plt.legend()

plt.show()


# In[20]:


def run_k_trials_with_time(k, p, n):
    trials = []
    for i in range(k):
        m = math.ceil((i+1)*n/k)
        trial = trial_with_time(i, m, n, p)
        trials.append(trial)
    return trials


# In[21]:


run_k_trials_with_time(5, 0.01, 20)


# In[22]:


def make_df(trials):
    df = pd.json_normalize(trials)
    df = df.drop(columns=["test_1.time","test_2.time"])
    df.columns = ['id','m','n','p','test1_num_episodes','test1_avg_time','test1_std_dev','test1_max','test1_min','test2_num_episodes','test2_avg_time','test2_std_dev','test2_max','test2_min']
    df = df.set_index('id')
    return df


# In[23]:


trials = run_k_trials_with_time(100, 0.01, 20)
df = make_df(trials)
df.head(10)


# In[24]:


sns.scatterplot(data=df, x="m", y="test1_avg_time", hue="m").set(title='Average Time of Lightbulb Failure: Test 1')


# In[25]:


sns.scatterplot(data=df, x="m", y="test2_avg_time", hue="m").set(title='Average Time of Lightbulb Failure: Test 2')


# In[33]:


trials = run_k_trials_with_time(100, 0.01, 20)

x = np.arange(100)
y1 = df.loc[:,"test1_avg_time"]
plt.scatter(x, y1, color=colors[0], label='Test 1')

y2 = df.loc[:,"test2_avg_time"]
plt.scatter(x, y2, color=colors[1], label='Test 2')

plt.xlabel('Sample id')
plt.ylabel('Avg. Time Between Failures')
plt.title('M-sample vs M:N-sample')
plt.legend()

plt.show()


# In[27]:


df.describe()


# In[28]:


import dataframe_image as dfi


# In[29]:


dfi.export(df.describe(),"dataframe.png")


# In[30]:


from pathlib import Path  
filepath = Path('./dataframe.csv')  
filepath.parent.mkdir(parents=True, exist_ok=True)  
df.to_csv(filepath)  


# In[31]:


def calc_harmonic(n):
    return sum(Fraction(1, d) for d in range(1, n + 1))


# In[32]:


# H_m
f = calc_harmonic(10)
h = round((f.numerator/f.denominator),4)
h


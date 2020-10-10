#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 22:59:20 2020

@author: jacopo
"""

'''
Code to create the variable 'availability', defined as the number of questions publishes still without an accepted answer
at any given day
'''

import pandas as pd
import classes as cl
import matplotlib.pyplot as plt
from collections import Counter
import re
import numpy as np
import matplotlib

#qa_name = 'apple/'
qa_name = 'ell/'

directory = '/home/jacopo/OneDrive/Dati_Jac_locali/stack/'
out_dir = directory + qa_name

synonyms_dict = pd.read_pickle(out_dir + 'tag_synonyms_dict.pkl') # from topics.py
topicsFromTags = pd.read_pickle(out_dir + 'topicsFromTags.pkl') # from topics.py

questions = cl.Questions(qa_name, directory, out_type='df').questions()
questions.loc[:,'CreationDate'] = questions['CreationDate'].apply(cl.date)
questions.loc[:,'CreationDate'] = questions['CreationDate'].apply(lambda x: pd.Timestamp(x.year,x.month,x.day))

# questions' topics
questions.loc[:,'TagList'] = questions['Tags'].apply(lambda x: re.findall('<(.+?)>',x))

def replaceSyn(x):
    newlist = []
    for i in x:
        if i in synonyms_dict.keys():
            newlist.append(synonyms_dict[i])
        else:
            newlist.append(i)
    return newlist

questions.loc[:,'TagList'] = questions['TagList'].apply(lambda x: replaceSyn(x))

topic_lists = [] # for each question get a counter of the topics found, 
for q_tags in questions['TagList'].tolist():
    newlist = []
    for i in q_tags:
        for topic in topicsFromTags.keys():
            if i in topicsFromTags[topic]:
                newlist.append(topic) # if tag not appears in any topics dict, then it's skipped and disregarded
    c = Counter(newlist)
    topic_lists.append(c)

topicdf = pd.DataFrame(topic_lists) 
topicdf.any(axis=1).sum() # 67517 questions got assigned at least 1 topic, out of 67522
topicdf.fillna(0, inplace=True)
topicdf = topicdf / np.sum(topicdf.values, axis=1)[:,np.newaxis]

questions = pd.concat([questions,topicdf], axis=1) 
questions = questions.loc[(questions[list(topicsFromTags.keys())].notna()).all(axis=1),] # drop questions not allocated to a topic

# questions with accepted answers
q_w_accanswer = questions.loc[~questions['AcceptedAnswerId'].isna()]

votes = cl.Votes(qa_name, directory, out_type='df').votes()
# answer acceptances (recover date)
acc = votes.loc[votes['VoteTypeId']=='1',]
acc.loc[:,'CreationDate'] = acc['CreationDate'].apply(cl.date)
acc.loc[:,'CreationDate'] = acc['CreationDate'].apply(lambda x: pd.Timestamp(x.year,x.month,x.day))
acc = acc.rename(columns={'CreationDate':'VoteDate', 'Id':'VoteId'})

# check that there is correspondance
m = pd.merge(q_w_accanswer, acc, left_on='AcceptedAnswerId', right_on='PostId', validate='1:1', indicator=True, how='outer')
m['_merge'].value_counts()
'''
APPLE
both          31268 (31271 but 3 dropped because no assignment of topic)
right_only      113 --> 110 likely to be deleted answers/questions, 3 dropped because no assignment of topic
left_only        34 --> seems some bug: i checked one case. The answer is selected as best answer, but it does not provide the time (so like the vote as not been saved)
Name: _merge, dtype: int64
ELL
both          39336
right_only      206
left_only         4
Name: _merge, dtype: int64
'''
acc = pd.merge(q_w_accanswer, acc, left_on='AcceptedAnswerId', right_on='PostId', validate='1:1', indicator=True, how='inner')

### FINAL CONSTRUCTION
topics = [i for i in topicsFromTags.keys()]

# questions with accepted answers for which we have date of acceptance
qa = acc[['CreationDate','Id']+topics]

# questions without an accepted answer
qnota = questions.loc[questions['AcceptedAnswerId'].isna(),['CreationDate','Id']+topics]

# all questions
q = pd.concat([qa,qnota])
q['numQ'] = 1

# new questions published per day

allq = q.groupby('CreationDate')[['numQ']+topics].sum()

# cumulative num of questions in platform
allq_cum = allq.cumsum()

# acceptance dates for existing questions that have an accepted answer
a = acc[['VoteDate','Id']+topics]
a['numQ'] = -1 # each time a question accepts an answer, it leaves the pool of available questions
a.loc[:,topics] = a[topics] * -1
a.rename(columns={'VoteDate':'CreationDate'}, inplace=True)
# cumulative number of questions still without an accepted answer
q2answer = pd.concat([q,a])
q2answer = q2answer.groupby('CreationDate')[['numQ']+topics].sum()
q2answer_cum = q2answer.cumsum()

pd.DataFrame(q2answer_cum).to_csv(out_dir + 'availability.csv') 

# plot: overall availability
plt.figure()
allq_cum['numQ'].plot(label='total')
q2answer_cum['numQ'].plot(label='still without an accepted answer',linestyle='--', color='black')
plt.legend(title='Cumulative number of questions:')
plt.xlabel('Date')
plt.ylabel('Num. Questions')

# saved as 'availability.png'

# plot: availability per topic
colors = matplotlib.cm.get_cmap(name='tab10')

fig = plt.figure()
ax = fig.gca()
for topic in topics:
    allq_cum[topic].plot(label='questions about "' + topic + '"', ax=ax, color=colors(topics.index(topic)))
    q2answer_cum[topic].plot(linestyle='--', label='questions about "' + topic + '" without an accepted answer', ax=ax, color=colors(topics.index(topic)))
plt.legend(title='Cumulative number of questions:')
plt.xlabel('Date')
plt.ylabel('Num. Questions')


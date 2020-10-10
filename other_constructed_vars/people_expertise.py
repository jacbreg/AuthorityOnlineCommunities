#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 18:33:00 2020

@author: jacopo
"""

'''
code to recover topics of experize of each author
'''

import pandas as pd
import classes as cl
import matplotlib.pyplot as plt
import re
from collections import Counter
import numpy as np

#qa_name = 'apple/'
qa_name = 'ell/'

directory = '/home/jacopo/OneDrive/Dati_Jac_locali/stack/'
out_dir = directory + qa_name

posts = cl.Posts(qa_name, directory, out_type='df').posts()
synonyms_dict = pd.read_pickle(out_dir + 'tag_synonyms_dict.pkl') # from topics.py
topicsFromTags = pd.read_pickle(out_dir + 'topicsFromTags.pkl') # from topics.py

answers = posts.loc[posts['PostTypeId']=='2',['Id','OwnerUserId','ParentId']]

questions = posts.loc[posts['PostTypeId']=='1',['Id','Tags']]
questions.rename(columns={'Id':'ParentId'},inplace=True)

answers = pd.merge(answers,questions,on='ParentId',how='inner',validate='m:1')


answers.loc[:,'TagList'] = answers['Tags'].apply(lambda x: re.findall('<(.+?)>',x))

def replaceSyn(x):
    newlist = []
    for i in x:
        if i in synonyms_dict.keys():
            newlist.append(synonyms_dict[i])
        else:
            newlist.append(i)
    return newlist

answers.loc[:,'TagList'] = answers['TagList'].apply(lambda x: replaceSyn(x))

topic_lists = [] # for each answer get a counter of the topics found, 
for answer_tags in answers['TagList'].tolist():
    newlist = []
    for i in answer_tags:
        for topic in topicsFromTags.keys():
            if i in topicsFromTags[topic]:
                newlist.append(topic) # if tag not appears in any topics dict, then it's skipped and disregarded
    c = Counter(newlist)
    topic_lists.append(c)

topicdf = pd.DataFrame(topic_lists) 
topicdf.any(axis=1).sum() # 123377 answers have been assigned at least 1 topic, out of 123385
topicdf.fillna(0, inplace=True)
topicdf = topicdf / np.sum(topicdf.values, axis=1)[:,np.newaxis]

answers = pd.concat([answers,topicdf], axis=1) 
answers = answers.loc[(answers[list(topicsFromTags.keys())].notna()).all(axis=1),] # drop answers not allocated to a topic

answers.drop(['Tags','TagList'],axis=1,inplace=True)

answers.to_csv(out_dir + 'answers2topics.csv',index=False) # save for use in other codes

topics = [i for i in topicsFromTags.keys()]
expertise = answers.groupby('OwnerUserId')[topics].mean()
expertise.reset_index(inplace=True)
expertise.to_csv(out_dir + 'expertise.csv',index=False)






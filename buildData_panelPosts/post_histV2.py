#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 19:48:41 2020

@author: jacopo
"""

'''
new version of post history panel data.

considers only answers.
'''

import pandas as pd
import classes as cl
import os
import nltk
from bs4 import BeautifulSoup
import re
from sklearn.metrics.pairwise import cosine_distances
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from markdown import markdown

stopwords = nltk.corpus.stopwords.words('english')

#qa_name = 'apple/'
qa_name = 'ell/'

directory = '/home/jacopo/OneDrive/Dati_Jac_locali/stack/'
out_dir = directory + qa_name

# # servers dirs
# out_dir = 'S:\\users\\jacopo\\Documents\\'
# directory = '\\\\tsclient\\jacopo\\OneDrive\\Dati_Jac_locali\\stack\\'
# qa_name = 'apple\\'

##############
# SET NUMBER OF DAYS MAX AFTER CREATION:
max_numdays = 90
##############

allanswers = cl.Answers(qa_name, directory, out_type='df').answers()

# make date only year-month-day
allanswers.loc[:,'CreationDate'] = allanswers['CreationDate'].apply(cl.date)
allanswers.loc[:,'CreationDate'] = allanswers['CreationDate'].apply(lambda x: pd.Timestamp(x.year,x.month,x.day))

### Preparing input: EDITS
# get ids of answers
answersIds = allanswers.loc[:,'Id']

# select history data of answers and clean 
edits = cl.Edits(qa_name, directory, out_type='df').edits() # 592507
edits.loc[:,'CreationDate'] = edits['CreationDate'].apply(cl.date)
edits.loc[:,'CreationDate'] = edits.loc[:,'CreationDate'].apply(cl.ymdhms) # drop microseconds TO MERGE WITH SUGGESTED EDITS
edits = edits.loc[edits['PostId'].isin(answersIds),]

# add info if edit was an approved suggestion
wassug = pd.read_csv(directory + qa_name + 'wassug.csv')
edits = edits.merge(wassug, on='RevisionGUID', how='left', validate='m:1')

# create approval date for suggested edits
edits.loc[edits['wassug']==1,'approval_date'] = edits['CreationDate']

# keep only edits that create of modify answers
edits = edits.loc[edits['PostHistoryTypeId'].isin(['8','5','2']),]

# load suggested edits, select edits to answers in sample only, clean
se = pd.read_pickle(directory + qa_name + 'allSuggestedEdits.pkl')
sedf = pd.DataFrame(se['items'])
sedf = sedf.loc[sedf['post_id'].isin(answersIds)]
sedf.loc[:,'creation_date'] = sedf['creation_date'].apply(cl.StackAPIdate)
sedf.loc[:,'approval_date'] = sedf['approval_date'].apply(cl.StackAPIdate)
sedf.loc[:,'rejection_date'] = sedf['rejection_date'].apply(cl.StackAPIdate)
sedf.loc[:,'post_id'] = sedf['post_id'].apply(str)
sedf.loc[:,'is_suggested'] = 1
sedf.loc[:,'PostHistoryTypeId'] = '5'
sedf = sedf.rename(columns={'post_id':'PostId'})
sedf = sedf.drop(labels=['post_type','suggested_edit_id'], axis=1)

# merge in suggested edits 
sedf_approved = sedf.loc[sedf['approval_date'].notna(),] 
edits_approved = edits.loc[edits['approval_date'].notna(),] 

m = pd.merge(edits_approved, sedf_approved, how='left', on=['PostId','approval_date', 'PostHistoryTypeId'], validate='1:1',indicator=True)

sedf_pending = sedf.loc[(sedf['approval_date'].isna()) & (sedf['rejection_date'].isna()),] # 0
sedf_rejected = sedf.loc[~sedf['rejection_date'].isna(),] # there may be too recent rejected edits
direct_edits = edits.loc[edits['approval_date'].isna(),] 

edits = pd.concat([m,sedf_pending,sedf_rejected,direct_edits])
edits = edits.reset_index()
edits = edits.drop(columns=['index'])

# fix 'is_suggested' variable 
edits.loc[edits['is_suggested'].isna(),'is_suggested'] = edits['wassug']
edits = edits.drop(columns=['wassug'])

# fix 'CreationDate' variable - for suggested edits, now CreationDate is when edit is suggested. Approval or rejection happen in the respective date columns
edits.loc[edits['is_suggested']==1,'CreationDate'] = edits['creation_date']
edits = edits.drop(columns=['creation_date'])

# fix UserId: add UserId from 'proposing_user' when possible for suggested and rejected edits
edits.loc[(edits['rejection_date'].notna()) & (edits['proposing_user'].notna()),'UserId'] = edits.loc[(edits['rejection_date'].notna()) & (edits['proposing_user'].notna()),'proposing_user'].apply(lambda x: str(x['user_id']))
edits.loc[edits['UserId'].isna(),'UserId'] = edits['UserDisplayName']

# add post's owner variable and edit-is-owner dummy
post2owner = allanswers.loc[:,['Id','OwnerUserId','OwnerDisplayName']]
post2owner.loc[post2owner['OwnerUserId'].isna(),'OwnerUserId'] = post2owner['OwnerDisplayName']
post2owner = post2owner.drop(columns=['OwnerDisplayName']).rename(columns={'Id':'PostId'})
edits = pd.merge(edits,post2owner, how='left', on='PostId', validate='m:1')
edits.loc[(edits['PostHistoryTypeId']=='2') & (edits['OwnerUserId'].isna()),'OwnerUserId'] = 'CommunityWiki'

edits.loc[edits['UserId'] == edits['OwnerUserId'],'editorISowner'] = 1
edits.loc[:,'editorISowner'] = edits.loc[:,'editorISowner'].fillna(0) # apparently few times post-owners edits on their own post are approved/rejected. Not sure why, maybe you can ask one of your edit to be reviewed.

# set editorISowner to 1 even if condition above is not met for edits of type 2
# (not really necessary as this edits are only used to get original value of text and then dropped)
edits.loc[(edits['PostHistoryTypeId']=='2') & (edits['editorISowner']==0),'editorISowner'] = 1

# ---> Note: this is data at edit level, 1956 Id is na though, these are the suggested edits not approved.

# create variable for most recent date and SORT
edits.loc[edits['approval_date'].notna(),'MostRecent'] = edits['approval_date']
edits.loc[edits['rejection_date'].notna(),'MostRecent'] = edits['rejection_date']
edits.loc[edits['MostRecent'].isna(),'MostRecent'] = edits['CreationDate']
edits.sort_values(by='MostRecent', inplace=True)

# normalize dates to DAY
edits.loc[edits['approval_date'].notna(),'approval_date'] = edits.loc[edits['approval_date'].notna(),'approval_date'].apply(lambda x: pd.Timestamp(x.year,x.month,x.day))
edits.loc[edits['CreationDate'].notna(),'CreationDate'] = edits.loc[edits['CreationDate'].notna(),'CreationDate'].apply(lambda x: pd.Timestamp(x.year,x.month,x.day))
edits.loc[edits['rejection_date'].notna(),'rejection_date'] = edits.loc[edits['rejection_date'].notna(),'rejection_date'].apply(lambda x: pd.Timestamp(x.year,x.month,x.day))
edits.loc[edits['MostRecent'].notna(),'MostRecent'] = edits.loc[edits['MostRecent'].notna(),'MostRecent'].apply(lambda x: pd.Timestamp(x.year,x.month,x.day))

# if CreationDate is NA (suggested edits not matched with suggested-edit data) use MostRecent
edits.loc[edits['CreationDate'].isna(),'CreationDate'] = edits['MostRecent']

if qa_name == 'apple/':
    download_date = pd.Timestamp(2017,6,10) 
elif qa_name == 'ell/':
    download_date = pd.Timestamp(2020,5,31) 

allanswers = allanswers.loc[allanswers['CreationDate']<=download_date]
answersIds = allanswers.loc[:,'Id']

# year variable
edits['year'] = edits['MostRecent'].apply(lambda x: x.year)

# load votes 
votes = cl.Votes(qa_name, directory, out_type='df').votes()
# make date only year-month-day
votes.loc[:,'CreationDate'] = votes['CreationDate'].apply(cl.date)
votes.loc[:,'CreationDate'] = votes['CreationDate'].apply(lambda x: pd.Timestamp(x.year,x.month,x.day))

### Build panel
# postid= '86'
answers_added_to_data = []


allcount = 0
for postid in answersIds:
    allcount += 1
    
    # drop strong outlier post
    if qa_name == 'apple/' and postid == '23660':
        continue
    
    if allcount % 1000 == 0:
        print('%d posts remaining'%(len(answersIds)-allcount))

    if postid in answers_added_to_data:
        continue
    
    editsp_extra = edits.loc[(edits['PostId']==postid) & (edits['PostHistoryTypeId'].isin(['2','5','8'])),] # with edit of body creation
    
    # rollbacks
    rollbacks = editsp_extra.loc[editsp_extra['PostHistoryTypeId']=='8',]
    rollbacks = rollbacks.groupby('MostRecent').agg({'editorISowner':['sum','count'] })
    rollbacks.columns = rollbacks.columns.droplevel(0)
    rollbacks = rollbacks.rename(columns={'sum':'editorISowner_rollback','count':'numrollback'})
    cols00 = ['editorISowner_rollback','numrollback']

    editsp = editsp_extra.loc[editsp_extra['PostHistoryTypeId']=='5',]
    # suggested edits approved
    suggappr = editsp.loc[(editsp['is_suggested']==1) & (editsp['rejection_date'].isna()),]
    suggappr = suggappr.groupby('approval_date').agg({'editorISowner':['sum','count'] })
    suggappr.columns = suggappr.columns.droplevel(0)
    suggappr = suggappr.rename(columns={'sum':'editorISowner_sA','count':'numedits_sA'})
    cols0 = ['editorISowner_sA','numedits_sA']
    '''
    sA == suggested and Approved
    sR == suggested and Rejected
    d == direct edits
    '''
    
    # suggested edits rejected (num of rejected edits allocated to day of rejection: MAKES SENSE???)
    suggrej = editsp.loc[(editsp['is_suggested']==1) & (editsp['rejection_date'].notna()),]
    suggrej = suggrej.groupby('rejection_date').agg({'editorISowner':['sum','count'] })
    suggrej.columns = suggrej.columns.droplevel(0)
    suggrej = suggrej.rename(columns={'sum':'editorISowner_sR','count':'numedits_sR'})
    cols1 = ['editorISowner_sR','numedits_sR']
    
    # direct edits
    directs = editsp.loc[(editsp['is_suggested']==0),]
    directs = directs.groupby('CreationDate').agg({'editorISowner':['sum','count'] })
    directs.columns = directs.columns.droplevel(0)
    directs = directs.rename(columns={'sum':'editorISowner_d','count':'numedits_d'})
    cols2 = ['editorISowner_d','numedits_d']
    
    # compute text measures
    text = editsp_extra.loc[editsp_extra['Text'].notna(),]  
    text.loc[:,'Text'] = text['Text'].apply(lambda x: re.sub('[<>]','',x)) # if < or > in text, possible that markdown function fails to convert properly to html
    text.loc[:,'Text'] = text['Text'].apply(lambda x: BeautifulSoup(markdown(x),parser='html'))
    
    text.loc[:,'length'] = text.loc[:,'Text'].apply(lambda x: len(re.findall(r'(?u)\b\w+\b',x.text.lower())))
    text.loc[:,'nostop'] = text.loc[:,'Text'].apply(lambda x: len([i for i in re.findall(r'(?u)\b\w+\b',x.text.lower()) if not i in stopwords]))    
    text.loc[:,'precision'] = text.loc[:,'nostop'] / text.loc[:,'length']
    text.loc[:,'numpict'] = text.loc[:,'Text'].apply(lambda x: len(x.find_all('img')))
    text.loc[:,'numlinks'] = text.loc[:,'Text'].apply(lambda x: len(x.find_all('a')))

    # (distance of each version from the original)
    vectorizer = CountVectorizer(token_pattern=r'(?u)\b\w+\b')    
    if any(text['length']!=0):
        X = vectorizer.fit_transform([i.text for i in text['Text'].tolist()])
        dist = cosine_distances(X)
        text.loc[:,'DistFromOrig'] = dist[0]
    else:
        text.loc[:,'DistFromOrig'] = 0 # if there's no text, set all to zeros        
    
    # get text measures after first edit in the day (for first day it means text at the creation time, as here creation is an edit)

    firsttext = pd.DataFrame(text.groupby('MostRecent')[['length','nostop','precision','DistFromOrig','numpict','numlinks']].first())
    cols3a = [i + 'F' for i in firsttext.columns]
    firsttext.columns = cols3a
    
    # get text measures after last edit in the day
    lasttext = pd.DataFrame(text.groupby('MostRecent')[['length','nostop','precision','DistFromOrig','numpict','numlinks']].last())
    cols3b = [i + 'L' for i in lasttext.columns]
    lasttext.columns = cols3b
    
    # add votes that generate points (up, down accepted)
    votep = votes.loc[(votes['PostId']==postid) & (votes['VoteTypeId'].isin(['1','2','3'])),]
    votep = votep.groupby(['CreationDate','VoteTypeId'])['PostId'].count().unstack()
    votep = votep.rename(columns={'1':'AcceptedAnsw','2':'numUpvotes', '3':'numDownvotes'})
    cols4 = ['AcceptedAnsw','numUpvotes','numDownvotes']
    
    # put together
    posthist = pd.concat([rollbacks, suggappr, suggrej, directs, firsttext, lasttext, votep], axis=1)

    # reindex with daily dates for 'max_numdays' days
    start = editsp_extra.loc[editsp_extra['PostHistoryTypeId']=='2', 'CreationDate'].iloc[0]
    
    end = start + pd.Timedelta('%d days'%max_numdays)
    if end >= download_date:
        last_date = download_date
    else:
        last_date = end
    
    dates = pd.date_range(start=start, end=last_date, freq='D')
    
    posthist = posthist.reindex(dates)

    # ffil nans on quality variables
    for col in cols3a+cols3b:
        posthist.loc[:,col] =  posthist.loc[:,col].fillna(method='ffill')

    # fill rest with zeros
    posthist = posthist.fillna(0)

    posthist.loc[:,'numedits_sAOthers'] = posthist['numedits_sA'] - posthist['editorISowner_sA']
    posthist.loc[:,'numedits_sROthers'] = posthist['numedits_sR'] - posthist['editorISowner_sR']
    posthist.loc[:,'numedits_dOthers'] = posthist['numedits_d'] - posthist['editorISowner_d']
    posthist.loc[:,'numedits_totalOthers'] = posthist['numedits_sAOthers'] + posthist['numedits_sROthers'] + posthist['numedits_dOthers'] 
    
    cols5 = ['numedits_sAOthers','numedits_sROthers','numedits_dOthers','numedits_totalOthers']

    # other vars
    posthist.loc[:,'periods'] = np.arange(0,len(posthist))
    posthist.loc[:,'PostId'] = postid
    posthist.loc[:,'OwnerUserId'] = editsp_extra.loc[editsp_extra['PostHistoryTypeId']=='2', 'OwnerUserId'].iloc[0]
    
    # id of the question that the answer is answering
    QuestionId = allanswers.loc[allanswers['Id']==postid,'ParentId'].values[0]
    posthist.loc[:,'ParentId'] = QuestionId
    
    colnames_fullset = ['PostId','periods','OwnerUserId'] + cols00 + cols0 + cols1 + cols2 + cols3a + cols3b + cols4 + cols5 + ['ParentId']
    posthist = posthist.reindex(colnames_fullset, axis='columns')
    posthist = posthist.fillna(0)
    
    posthist.index.name = 'day'
    posthist.reset_index(inplace=True)
    
    if len(posthist)>max_numdays+1:
        print('PROBLEM: TOO MANY DAYS')
        break
    
    if 'postHistV2.csv' in os.listdir(out_dir):
        posthist.to_csv(out_dir + 'postHistV2.csv', mode='a', header=False, index=False)
    else:
        posthist.to_csv(out_dir + 'postHistV2.csv', mode='a', index=False)
    answers_added_to_data.append(postid)
        
# save data for period 1 only
df = pd.read_csv(out_dir + 'postHistV2.csv',dtype={'OwnerUserId': 'str'})
df1period = df[(df['periods']==0)]
df1period.to_csv(out_dir + 'postHistV2_1period.csv', index=False)

# save data for stata
df = pd.read_csv(out_dir + 'postHistV2.csv',dtype={'OwnerUserId': 'str'}, parse_dates=['day'])
# transform OwnerUserId to numeric for readability in stata (TO NOT MERGE WITH OTHER USER DATA - JUST USABLE FOR FIXED EFFECTS, CLUSTER STD ERR AT USER LEVEL ECC)
count = 0
owner2num = {} 
for owner in df.OwnerUserId.unique():
    owner2num[owner] = count
    count += 1

df.loc[:,'OwnerUserId'] = df['OwnerUserId'].apply(lambda x: owner2num[x])
df.to_stata(out_dir + 'postHistV2.dta', write_index=False)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 19:41:57 2019

@author: jacopo
"""

'''
create panel of user histories with the amount of quality - do of both answers and edits
 - THIS VERSION CONSIDERS ONLY ANSWERS AND EDITS OVER ANSWERS
 - Assumed that 3 months of no activity after last action determin exit from the platform
'''

import pandas as pd
import classes as cl
import os
import nltk
from joblib import Parallel, delayed
import multiprocessing
import numpy as np
import re
from bs4 import BeautifulSoup
from markdown import markdown
import seaborn as sns
import matplotlib.pyplot as plt

stopwords = nltk.corpus.stopwords.words('english')

# qa_name = 'apple/'
qa_name = 'ell/'

directory = '/home/jacopo/OneDrive/Dati_Jac_locali/stack/'
reputation_dir = directory + qa_name + 'reputation/'
out_dir = directory + qa_name #+  'userHist/'

# servers dirs
out_dir = 'S:\\users\\jacopo\\Documents\\UserHist\\'
directory = '\\\\tsclient\\jacopo\\OneDrive\\Dati_Jac_locali\\stack\\'
qa_name = 'apple\\'
reputation_dir = directory + qa_name + 'reputation\\'

### get list of users that have published at least 1 answer
allanswers = cl.Answers(qa_name, directory, out_type='df').answers()

answerers = allanswers.loc[allanswers['OwnerUserId'].notna(),'OwnerUserId'].unique().tolist()

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

if qa_name == 'apple/':
    download_date = pd.Timestamp(2017,6,10) 
elif qa_name == 'ell/':
    download_date = pd.Timestamp(2020,5,31) 

# questions, to see if community answered own questions
questions = cl.Questions(qa_name, directory, out_type='df').questions()
# make date only year-month-day
questions.loc[:,'CreationDate'] = questions['CreationDate'].apply(cl.date)
questions.loc[:,'CreationDate'] = questions['CreationDate'].apply(lambda x: pd.Timestamp(x.year,x.month,x.day))

# year variable
edits['year'] = edits['MostRecent'].apply(lambda x: x.year)


# load votes for altruism variable
votes = cl.Votes(qa_name, directory, out_type='df').votes()
# keep only upvotes and downvotes and accepted answers
votes = votes.loc[votes['VoteTypeId'].isin(['1','2','3'])]
# make date only year-month-day
votes.loc[:,'CreationDate'] = votes['CreationDate'].apply(cl.date)
votes.loc[:,'CreationDate'] = votes['CreationDate'].apply(lambda x: pd.Timestamp(x.year,x.month,x.day))

# load registration date for seniority variable 
users = cl.Users(qa_name, directory, out_type='df').users()
users.loc[:,'CreationDate'] = users['CreationDate'].apply(cl.date)
users.loc[:,'CreationDate'] = users['CreationDate'].apply(lambda x: pd.Timestamp(x.year,x.month,x.day))
users = users[['Id','CreationDate']]
users.rename(columns={'CreationDate':'RegistrationDate'},inplace=True)

# model: measure of quality
modfit = pd.read_pickle(directory + qa_name + 'qualityfit.pkl')

if qa_name == 'apple/':
    # list of users for whome I have reputation data
    listdir = os.listdir(reputation_dir)
    repfiles = [i for i in listdir if not os.stat(reputation_dir + i).st_size < 4]
    repfilesIDs = [re.findall('[0-9]+',i)[0] for i in repfiles]
    answerers = [i for i in answerers if i in repfilesIDs]

elif qa_name == 'ell/':
    repdata = pd.read_csv(directory + qa_name + 'ELLrep.csv', parse_dates=['date'], dtype={'user':str})
    repdata.set_index('date', inplace=True)
    usersrep = repdata['user'].unique().tolist()
    answerers = [i for i in answerers if i in usersrep]

# thresholds
thresholds = pd.read_csv(directory + qa_name + 'thresholds.csv')
thresholds = thresholds.loc[(thresholds['type'].notna()) & (thresholds['type']!='Communication')] # some thresholds are not displayed + communiaction thresolds are more complicated: see https://meta.stackexchange.com/questions/58587/what-are-the-reputation-requirements-for-privileges-on-sites-and-how-do-they-di
thresholds = thresholds.loc[thresholds['rep-level']>10] # remove 1 and 10 thresholds
Tgrad = sorted(thresholds['rep-level'].unique().tolist()[::-1])
Tbeta = sorted(thresholds['rep-level-publicbeta'].unique().tolist()[::-1])

if qa_name == 'ell/':
    grad_announced = pd.Timestamp(2015,9,10) 
    grad_rep = pd.Timestamp(2016,2,25) 
    
#user = '5472'
#user = '24231'
for user in answerers:
        
    cols_to_merge = []
    
    if qa_name == 'apple/':
        try:
            rep = pd.read_csv(reputation_dir + user + '.csv', index_col = 1, usecols=[1,2], parse_dates=True)
        except:
            continue
    else:
        rep = repdata.loc[repdata['user']==user,'rep']
    
    cols_to_merge.append(rep)
    
    # answer ids
    user_answers = allanswers.loc[allanswers['OwnerUserId']==user,'Id']
    # history of users' answers (remove rejected edits)
    user_edits = edits.loc[(edits['PostId'].isin(user_answers)) & (edits['rejection_date'].isna()),]
    
    # remove edits made by '-1' not suggested: this are prob. authomatic edits since 
    # community-owned posts by construction are not included in user_edits.
    # (anonymous suggestions are also owned by -1)
    user_edits = user_edits.loc[~((user_edits['UserId']=='-1') & (user_edits['is_suggested']==0)),]
          
    ### CHOICES ##################################################################################
    
    # quality: consider only text at creation date, excluding all edits. Mean across answers in the same day
    qual = user_edits.loc[user_edits['PostHistoryTypeId']=='2']
    qual.loc[:,'Text'] = qual['Text'].apply(lambda x: re.sub('[<>]','',x))
    qual.loc[:,'Text'] = qual['Text'].apply(lambda x: BeautifulSoup(markdown(x),parser='html'))
    qual.loc[:,'lengthF'] = qual.loc[:,'Text'].apply(lambda x: len(re.findall(r'(?u)\b\w+\b',x.text.lower()))) # regex inspired by regex used by sklearn.feature_extraction.text.CountVectorizer
    qual.loc[:,'nostop'] = qual.loc[:,'Text'].apply(lambda x: len([i for i in re.findall(r'(?u)\b\w+\b',x.text.lower()) if not i in stopwords]))    
    qual.loc[:,'precisionF'] = qual.loc[:,'nostop'] / qual.loc[:,'lengthF']
    qual.loc[:,'numpictF'] = qual.loc[:,'Text'].apply(lambda x: len(x.find_all('img')))
    qual.loc[:,'numlinksF'] = qual.loc[:,'Text'].apply(lambda x: len(x.find_all('a')))
    
    qual.loc[:,'precisionF2'] = qual['precisionF']**2
    qual.loc[:,'lengthF2'] = qual['lengthF']**2
    qual.loc[:,'numpictF2'] = qual['numpictF']**2
    qual.loc[:,'numlinksF2'] = qual['numlinksF']**2

    qual.loc[:,'quality'] = modfit.predict(qual[['precisionF','precisionF2','lengthF','lengthF2','numpictF','numpictF2','numlinksF','numlinksF2']])

    qual = qual.groupby('MostRecent')['quality'].mean() # i pick the mean since recording the quantity too, the total sum is recoverable
    
    cols_to_merge.append(qual)
    
    # quantity
    numanswers = user_edits.loc[user_edits['PostHistoryTypeId']=='2',].groupby(['MostRecent'])['PostId'].count().rename('numAnswers')
    cols_to_merge.append(numanswers)
    
    # edits
    onothersdf = edits.loc[(edits['UserId']==user) & (edits['editorISowner']==0),]
    if len(onothersdf)!=0:
        onothersdf.loc[onothersdf.approval_date.notna(),'is_approved'] = 1
        onothersdf.loc[:,'is_approved'].fillna(0,inplace=True)
        onothers = onothersdf.groupby(['CreationDate','is_suggested'])['is_approved'].agg(['count','sum'])
        onothers = onothers.unstack()
        onothers.fillna(0,inplace=True)
        onothers.columns = [i[0] + str(int(i[1])) for i in onothers.columns]
        onothers.rename(columns={"count0":'Ed',"count1":'Es',"sum1":'Esa'}, inplace=True)
        if 'sum0' in onothers.columns:
            onothers.drop(columns='sum0', inplace=True)        
        cols_to_merge.append(onothers)
        
    # Remark: Edits made on own content are not saved
    
    ### EXPERIENCE ###############################################################################
    
    # experience: AnswerNum = num of answers published BEFORE the given day (SHIFTED BELOW) 
    answernum = user_edits.loc[user_edits['PostHistoryTypeId']=='2']
    answernum.loc[:,'AnswerNum'] = np.arange(1,len(answernum)+1, step=1)
    answernum = answernum.groupby('MostRecent')['AnswerNum'].last()
    cols_to_merge.append(answernum)
    
    # Seniority is made at the end
    
    ### RECIPROCITY ##############################################################################

    # num of received edits + received rollbacks
    user_edits.loc[user_edits.approval_date.notna(),'is_approved'] = 1
    user_edits.loc[:,'is_approved'].fillna(0,inplace=True)
    numEdits = user_edits.loc[(user_edits['editorISowner']==0),].groupby(['MostRecent','is_suggested'])['is_approved'].agg(['count','sum']) 
    numEdits = numEdits.unstack()
    numEdits.fillna(0,inplace=True)
    numEdits.columns = [i[0] + str(int(i[1])) for i in numEdits.columns]
    numEdits.rename(columns={"count0":'EOd',"count1":'EOs',"sum1":'EOsa'}, inplace=True)
    if 'sum0' in numEdits.columns:
        numEdits.drop(columns='sum0', inplace=True)    # direct edits are not approved by construction, so the column would be only zeros
    if any(numEdits):
        cols_to_merge.append(numEdits)

    # num of received answers on owned questions
    user_questions_ids = questions.loc[questions['OwnerUserId']==user,'Id']
    answers_toUserQ = allanswers.loc[allanswers['ParentId'].isin(user_questions_ids)]
    # recover, for each question, the first, second,.. answer
    answers_toUserQ['order'] = 1
    answers_toUserQ = answers_toUserQ.sort_values(by='CreationDate')
    answers_toUserQ.loc[:,'order'] = answers_toUserQ.groupby('ParentId')['order'].transform(np.cumsum)
    #total number of answers to own questions
    answers_toUserQ_all = answers_toUserQ.groupby('CreationDate')['Id'].count().rename('received_answers')
    # number of first answers to own questions
    answers_toUserQ_first = answers_toUserQ.loc[answers_toUserQ['order']==1,].groupby('CreationDate')['Id'].count().rename('received_1st_answers')
    if any(answers_toUserQ_all):
        cols_to_merge.append(answers_toUserQ_all)
        cols_to_merge.append(answers_toUserQ_first)
    
    
    # record num of questions made as well (for beliefs on arrival of answers)
    user_questions = questions.loc[questions['OwnerUserId']==user,]
    user_questions = user_questions.groupby('CreationDate')['Id'].count().rename('numQuestions')
    if any(user_questions):
        cols_to_merge.append(user_questions)
    
    ### ALTRUISM #################################################################################
    
    # in answering- num of answers selected as best answers
    answ_ids = user_edits.loc[user_edits['PostHistoryTypeId']=='2','PostId']
    votes_receveid = votes.loc[(votes['PostId'].isin(answ_ids)) & (votes['VoteTypeId']=='1')]
    votes_receveid = votes_receveid.groupby('CreationDate')['Id'].count().rename('numAcceptedAnswers')
    if any(votes_receveid):
        cols_to_merge.append(votes_receveid)
        
    # in editing
    if len(onothersdf)!=0:
        # base variable for altruism variable: points made by answers of others edited by user
        implemented = onothersdf.loc[(onothersdf['is_suggested']==0) | (onothersdf['is_approved']==1),['PostId','MostRecent']]
        implemented = implemented.groupby('PostId')['MostRecent'].first()

        othersVotes = pd.merge(votes, implemented, left_on='PostId', right_index=True,
                               how='inner') # , validate='m:1' --> correct. not done for speed
        if len(othersVotes)!=0:
            othersVotes = othersVotes.loc[othersVotes['CreationDate']>=othersVotes['MostRecent']] # only votes after edit occurred
            othersVotes = othersVotes.groupby(['CreationDate','VoteTypeId'])['Id'].count()
            othersVotes = othersVotes.unstack().fillna(0)
            
            namedict = {'1':'EditedPostIsAccepted','2':'EditedPostsUpVotes','3':'EditedPostsDownvotes'}
            othersVotes.columns = [namedict[i] for i in othersVotes.columns]
            
            cols_to_merge.append(othersVotes)
        
    hist = pd.concat(cols_to_merge, axis=1) 

    # set last-day as download date if reputation points data gets there
    # or 3 months after max(last rep obtained, last action made)
    lastdta = max(hist.index.sort_values())
    if lastdta + pd.Timedelta(90,unit='D') >= download_date:
        last_date = download_date
    else:
        last_date = lastdta + pd.Timedelta(90,unit='D')
    first_date = min(hist.index.sort_values()) # period 0 is start of actions in platform, not creation of account
    
    index = pd.date_range(start=first_date, end=last_date, freq='D')
    
    hist = hist.reindex(index)
            
    # periods
    hist.loc[:,'periods'] = np.arange(0,len(hist))

    # user
    hist.loc[:,'user'] = user
    
    hist.fillna(0,inplace=True)

    colnames_fullset = ['periods','user','rep','rep_cum','quality','numAnswers','numQuestions',
                        'Ed','Es','Esa','AnswerNum','Seniority_days','EOd','EOs','EOsa',
                        'received_answers','received_1st_answers','numAcceptedAnswers','EditedPostIsAccepted',
                        'EditedPostsUpVotes','EditedPostsDownvotes']
    hist = hist.reindex(colnames_fullset, axis='columns')
    hist.fillna(0,inplace=True)

    # rep cum
    hist.loc[:,'rep_cum'] = hist['rep'].cumsum()

    # 1 in day of passing all thresholds
    hist.loc[:,'Tcum'] = 0
    for t in Tbeta:
        hist.loc[(hist.index < grad_rep) & (hist['rep_cum']>=t),'Tcum'] = 1 + hist.loc[(hist.index < grad_rep) & (hist['rep_cum']>=t),'Tcum']
    for t in Tgrad:
        hist.loc[(hist.index >= grad_rep) & (hist['rep_cum']>=t),'Tcum'] = 1 + hist.loc[(hist.index >= grad_rep) & (hist['rep_cum']>=t),'Tcum']
    
    # experience: Seniority = num of days the users has been registered in the platform at the prvious period 
    reg = users.loc[users['Id']==user,'RegistrationDate']
    if len(reg)>1:
        print('problem----> not unique match of user id')
        break

    hist.loc[:,'Seniority_days'] = hist.index - reg.iloc[0] 
    hist.loc[:,'Seniority_days'] = hist['Seniority_days'].apply(lambda x: x.days) 
    hist.loc[hist['Seniority_days']<0,'Seniority_days'] = 0 # set to zero when negative (negative is there are obs before private beta)

    if 'UserHist.csv' in os.listdir(out_dir):
        hist.to_csv(out_dir + 'UserHist.csv', mode='a', header=False) 
    else:
        hist.to_csv(out_dir + 'UserHist.csv', mode='a')     

###################################################################################################################
hist = pd.read_csv(out_dir + 'UserHist.csv', index_col=0, parse_dates=True)
hist.index.name = 'day'
hist.reset_index(inplace=True)


###################################################################################################################
#### add availability variable
###################################################################################################################

availability = pd.read_csv(out_dir + 'availability.csv',index_col=0, parse_dates=True )
expertise = pd.read_csv(out_dir + 'expertise.csv')

# add all dates to availability
compl_avail_index = pd.date_range(start=availability.index.min(), end=availability.index.max(), freq='D')
availability = availability.reindex(compl_avail_index, method='ffill')
availability.rename(columns=lambda x: 'avail_'+x, inplace=True)

# merge availability
hist = pd.merge(hist, availability, how='left', left_on='day', right_index=True, indicator=True)
'''
hist['_merge'].value_counts()
both          20507627
right_only           0
left_only            0
Name: _merge, dtype: int64
'''
# rename expertise columns
expertise.rename(columns={'OwnerUserId':'user'}, inplace=True)

# merge expertise
hist = pd.merge(hist, expertise, how='left', on='user', indicator='_merge2')
'''
hist['_merge2'].value_counts()
both          20507627
right_only           0
left_only            0
Name: _merge2, dtype: int64
'''

# create variable
avail = np.repeat(0.0, len(hist))
topics = [i for i in expertise.columns if i!='user']
avtopics = ['avail_'+i for i in topics]
for pair in list(zip(avtopics,topics)):
    avail += np.multiply(hist[pair[0]].values,hist[pair[1]].values) # multiply availability X expertise and sum across topics
hist.loc[:,'avail'] = avail

hist.drop(availability.columns.tolist() + topics + ['_merge','_merge2'], axis=1, inplace=True)

# hist.loc[:,'avail'] = (hist['avail_macos'] * hist['macos']) + (hist['avail_iphone'] * hist['iphone']) + (hist['avail_ios'] * hist['ios']) + (hist['avail_macbook-pro'] * hist['macbook-pro'])


hist.sort_values(by=['user','periods'], inplace=True)

# plot
sns.lineplot(x='periods',y='avail', data=hist, ci='sd')
plt.xlabel('Periods of participation')
plt.ylabel('Experience-based Availability')
# saved as availxtime.png

#################################################################################################################
### choice of quality and quantity of answers +  construction of lambda (for week level see below)
#################################################################################################################
print('start lambda')
## FUNCTIONS, MODELS, PARAMETERS
# function: decay of points
def expo(t, A, tau): # exponential decay function; A = aplitude, tau = decay time
	return A * np.exp(-t/tau) 
# model: expected accepted edits
EAE = pd.read_pickle(directory + qa_name + 'PoissonReg_EdvsQ_noTopics.pkl')
# model: expected upvotes
EUV = pd.read_pickle(directory + qa_name + 'PoissonReg_UpvsQ_noTopics.pkl')
# model: expected downvotes
EDV = pd.read_pickle(directory + qa_name + 'PoissonReg_DownvsQ_noTopics.pkl')
# parameter: tau
tau_up = pd.read_pickle(directory + qa_name + 'decay_params_up.pkl')[1] # first num is estim of A, second of tau
tau_down = pd.read_pickle(directory + qa_name + 'decay_params_down.pkl')[1] # first num is estim of A, second of tau
##

# load choices' bins' boundaries
quality_bins = pd.read_pickle(out_dir + 'choice_qualityansw_bins.pkl')
quantity_bins = pd.read_pickle(out_dir + 'choice_quantityansw_bins.pkl')

# load choice values
quality_choices = pd.read_pickle(out_dir + 'choice_qualityansw.pkl')
leftbound_quality = [0] + [i.left for i in quality_choices.index] + [quality_choices.index[-1].right]
quantity_choices = pd.read_pickle(out_dir + 'choice_quantityansw.pkl')
leftbound_quantity = [0] + [i.left for i in quantity_choices.index] + [quantity_choices.index[-1].right]

# negative values of quality (6 obs in apple, 3 in ell) forced to be in the first quality level
hist.loc[hist['quality']<0,'quality']= 0
# bin quality variable, creating the choice variable
hist.loc[:,'qualitybins'] = pd.cut(hist['quality'], bins=leftbound_quality, labels=[0] + quality_choices.tolist(), right=True, include_lowest=True)

# bin quantity variable
hist.loc[:,'quantitybins'] = pd.cut(hist['numAnswers'], bins=leftbound_quantity, labels=[0] + quantity_choices.tolist(), right=True, include_lowest=True)

# change names to fit with saved models' variables
hist.rename(columns={'quality':'quality_orig','numAnswers':'numAnswers_orig','qualitybins':'quality','quantitybins':'numAnswers'}, inplace=True)
# expected number of edits received on new answers given 1) previous period realization of states, 2) choices
hist.loc[:,'numedits_totalothers_accepted'] = EAE.predict(hist[['quality','AnswerNum','Seniority_days']])   

# expected number of upvotes/downvotes received on new answers given 1) previous period realization of states, 2) choices
# upvotes 
''' WROOONG SHOULD BE NUM-ANSWERS X EUV - see below for week version '''  
hist.loc[hist['numAnswers']>0,'EUV'] = EUV.predict(hist[['quality','AnswerNum','Seniority_days','numedits_totalothers_accepted']])
hist.loc[hist['numAnswers']==0,'EUV'] = 0
# downvotes
hist.loc[hist['numAnswers']>0,'EDV'] = EDV.predict(hist[['quality','AnswerNum','Seniority_days','numedits_totalothers_accepted']])
hist.loc[hist['numAnswers']==0,'EDV'] = 0

def lambda_f(x, tau):
    '''
    x  := vector of per period values of A
    ''' 
    time_matrix = np.tile(np.arange(0,len(x)),(len(x),1))
    time_matrix = np.maximum(time_matrix - np.arange(0,len(x))[:,np.newaxis], np.zeros((len(x),len(x))))
    means = expo(time_matrix,x.values[:,np.newaxis],tau)
    means = np.triu(means,1)
    means = np.sum(means,axis=0)
    return means

# make sure periods are sorted
hist.sort_values(by=['user','periods'], inplace=True)

# expected number of upvotes arriving from past actions
hist.loc[:,'lambda_up'] = hist.groupby('user')['EUV'].transform(lambda x: lambda_f(x,tau_up))

# expected number of downvotes arriving from past actions
hist.loc[:,'lambda_down'] = hist.groupby('user')['EDV'].transform(lambda x: lambda_f(x,tau_down))

#################################################################################################################
### choice of number of edits
#################################################################################################################
print('start edits')
# fix edit choice variable
hist.loc[:,'numEdits_orig'] = hist['Ed'] + hist['Es']
edits_choices = pd.read_pickle(out_dir + 'choice_quantityedits.pkl')
leftbound_edits = [0] + [i.left for i in edits_choices.index] + [edits_choices.index[-1].right]
hist.loc[:,'numEdits'] = pd.cut(hist['numEdits_orig'], bins=leftbound_edits, labels=[0] + edits_choices.tolist(), right=True, include_lowest=True)

#hist.drop(['EP','lambda'], axis=1, inplace=True) # drop wrong columns 

# tot edits received
hist['EO'] = hist['EOd'] + hist['EOs']

# tot edits received implemented
hist['EOimpl'] = hist['EOd'] + hist['EOsa']

#################################################################################################################
### Ranking in platform
#################################################################################################################

temp = hist.pivot(index='user', columns='day', values='rep_cum')   
temp.fillna(method='ffill', axis=1, inplace=True)

for day in temp.columns:
    temp = temp.sort_values(by=day, ascending=False)
    a = temp[day].diff(1)
    a.iloc[0] = -1
    a = np.where(a<0,1,0)
    a = a.cumsum()
    temp.loc[:,day] = a

temp = temp.stack()
temp.name = 'rank'
hist = pd.merge(hist, temp, on=['user','day'], validate='1:1', how='left') # this adds the rank, 1 being the top

# recover number of users registered at each
allusers = cl.Users(qa_name, directory, out_type='df').users()
allusers.loc[:,'CreationDate'] = allusers['CreationDate'].apply(cl.date)
allusers.loc[:,'CreationDate'] = allusers['CreationDate'].apply(lambda x: pd.Timestamp(x.year,x.month,x.day))
numusers = allusers.groupby('CreationDate')['Id'].count() # does not consider users that are excluded from the platform
numusers.name = 'numUsers'
numusers = numusers.cumsum()

hist = pd.merge(hist, numusers, left_on='day', right_index=True, how='left',validate='m:1') # this add the num of users registered in the paltform
hist.loc[:,'topPercent'] = hist['rank'] / hist['numUsers']

# hist.sort_values(by=['user','day'], inplace=True)

# hist.loc[:,'varRank'] = hist.groupby('user')['rank'].transform(lambda x: x.diff(1))
# hist.loc[:,'varRankL'] = hist.groupby('user')['varRank'].transform(lambda x: x.shift(1))

# hist.loc[:,'rankL'] = hist.groupby('user')['rank'].transform(lambda x: x.shift(1))

# import statsmodels.formula.api as smf
# fit = smf.ols('varRank ~ rankL + C(numAnswers) + C(quality) + varnumUsers', data=hist).fit()

#################################################################################################################
### dummy for whether in given period user reached one or more thresholds
#################################################################################################################
hist.sort_values(by=['user','day'], inplace=True)

hist.loc[:,'ReachedT'] = hist.groupby('user')['Tcum'].transform(lambda x: x.diff(1))
hist.loc[:,'ReachedT'] = np.where(hist['ReachedT']>=1, 1, 0) # set to zero also negative variations. disutilities of loosing priv captured by Tcum

print('start save')
hist.to_csv(out_dir + 'UserHist_complete.csv', index=False)

### small - estimation specific data

hist = pd.read_csv(out_dir + 'UserHist_complete.csv', parse_dates=['day'])
# public beta day of site 
if qa_name == 'apple/':
    pb = pd.Timestamp(2010,8,24)
elif qa_name == 'ell/':
    pb = pd.Timestamp(2013,1,30)
# drop obs before public beta (still experience vars ecc don't neglect it, i just don't want to use it for inference on choices)
hist = hist.loc[hist['day']>=pb]

hist.sort_values(by=['user','day'], inplace=True)
# lag state variables

# experience: AnswerNum (for Seniority no shift because on first day of registration Seniority==0)
hist.loc[:,'AnswerNumL'] = hist.groupby('user')['AnswerNum'].transform(lambda x: x.shift(1,fill_value=0))
# rep cum
hist.loc[:,'rep_cumL'] = hist.groupby('user')['rep_cum'].transform(lambda x: x.shift(1,fill_value=0))
# reciprocity
hist.loc[:,'EOL'] = hist.groupby('user')['EO'].transform(lambda x: x.shift(1,fill_value=0))
hist.loc[:,'EOimplL'] =  hist.groupby('user')['EOimpl'].transform(lambda x: x.shift(1,fill_value=0))
hist.loc[:,'received_answersL'] =  hist.groupby('user')['received_answers'].transform(lambda x: x.shift(1,fill_value=0))
hist.loc[:,'received_1st_answersL'] =  hist.groupby('user')['received_1st_answers'].transform(lambda x: x.shift(1,fill_value=0))
# altruism
hist.loc[:,'numAcceptedAnswersL'] = hist.groupby('user')['numAcceptedAnswers'].transform(lambda x: x.shift(1,fill_value=0))
hist.loc[:,'EditedPostIsAcceptedL'] = hist.groupby('user')['EditedPostIsAccepted'].transform(lambda x: x.shift(1,fill_value=0))
hist.loc[:,'EditedPostsUpVotesL'] = hist.groupby('user')['EditedPostsUpVotes'].transform(lambda x: x.shift(1,fill_value=0))
hist.loc[:,'EditedPostsDownvotesL'] = hist.groupby('user')['EditedPostsDownvotes'].transform(lambda x: x.shift(1,fill_value=0))
# rank
hist.loc[:,'rankL'] = hist.groupby('user')['rank'].transform(lambda x: x.shift(1,fill_value=0))
hist.loc[:,'topPercentL'] = hist.groupby('user')['topPercent'].transform(lambda x: x.shift(1,fill_value=0))
# questions made
hist.loc[:,'numQuestionsL'] = hist.groupby('user')['numQuestions'].transform(lambda x: x.shift(1,fill_value=0))
# thresholds reached
hist.loc[:,'TcumL'] = hist.groupby('user')['Tcum'].transform(lambda x: x.shift(1,fill_value=0))
hist.loc[:,'ReachedTL'] = hist.groupby('user')['ReachedT'].transform(lambda x: x.shift(1,fill_value=0))

# # change reciprocity abd altruism variable to cumulative
# hist.loc[:,'received_answers_cum'] = hist.groupby('user')['received_answers'].cumsum()
# hist.loc[:,'EOimpl_cum'] = hist.groupby('user')['EOimpl'].cumsum()
# hist.loc[:,'numAcceptedAnswers_cum'] = hist.groupby('user')['numAcceptedAnswers'].cumsum()
# hist.loc[:,'EditedPostsUpVotes_cum'] = hist.groupby('user')['EditedPostsUpVotes'].cumsum()

# # scale EditedPostsUpVotes_cum and numAcceptedAnswers_cum
# hist.loc[:,'EditedPostsUpVotes_cum'] = np.maximum(0,np.log(hist['EditedPostsUpVotes_cum']))
# hist.loc[:,'numAcceptedAnswers_cum'] = np.maximum(0,np.log(hist['numAcceptedAnswers_cum']))

# hist.loc[:,'numQuestions_cum'] = hist.groupby('user')['numQuestions'].cumsum()

essential_cols = ['day','periods', 'user', 'rep_cumL', 'AnswerNumL', 'Seniority_days',
       'EOimplL', 'EOL', 'received_answersL','numQuestionsL','numAcceptedAnswersL',
       'EditedPostIsAcceptedL', 'EditedPostsUpVotesL', 'EditedPostsDownvotesL','TcumL',
       'ReachedTL','rankL','topPercentL','avail', 'quality', 'numAnswers', 'lambda_up', 
       'lambda_down', 'numEdits']

hist[essential_cols].to_csv(out_dir + 'UserHist_Csmall.csv', index = False)

#################################################################################################################
###################################### WEEKLY AGGREGATION #######################################################
#################################################################################################################

hist = pd.read_csv(out_dir + 'UserHist_complete.csv', parse_dates=['day'])

# recover total quality so to average on the week
hist.loc[:,'quality_orig'] = hist['quality_orig'] * hist['numAnswers_orig']

aggdict = {'rep_cum':'last',
           'periods':'last',
           'AnswerNum':'last',
           'Seniority_days':'last',
           'Tcum':'last',
           'avail':'mean',
           'quality_orig':'sum',
           'numAnswers_orig':'sum',
           'numEdits_orig':'sum',
           'numQuestions':'sum',
           'EOd':'sum',
           'EOs':'sum',
           'EOsa':'sum',
           'received_answers':'sum',
           'received_1st_answers':'sum',
           'numAcceptedAnswers':'sum',
           'EditedPostIsAccepted':'sum',
           'EditedPostsUpVotes':'sum',
           'EditedPostsDownvotes':'sum',
           'rep':'count'} # just to have number of days in each period after aggregation
hist = hist.groupby(['user', pd.Grouper(key='day',freq='W',closed='right', label='right')]).agg(aggdict)
hist = hist.reset_index()
# re-set quality as average
hist.loc[hist['numAnswers_orig']>0,'quality_orig'] = hist['quality_orig'] / hist['numAnswers_orig'] 

# remove periods composed by less than 7 days (should be initial period when period 0 is not on monday or last periods where last day is not sunday)
hist = hist.loc[hist['rep']==7]
hist.drop(columns='rep', inplace=True)
# create period variable
hist.loc[:,'periods'] = hist['periods'] // 7

#################################################################################################################
### choice of quality and quantity of answers +  construction of lambda
#################################################################################################################
 
# model: expected accepted edits
EAE = pd.read_pickle(directory + qa_name + 'PoissonReg_EdvsQ_noTopics.pkl')
# model: expected upvotes
EUV = pd.read_pickle(directory + qa_name + 'PoissonReg_UpvsQ_noTopics.pkl')
# model: expected downvotes
EDV = pd.read_pickle(directory + qa_name + 'PoissonReg_DownvsQ_noTopics.pkl')
# parameter: tau
tau_up = pd.read_pickle(directory + qa_name + 'decay_params_up_week.pkl')[1] # first num is estim of A, second of tau
tau_down = pd.read_pickle(directory + qa_name + 'decay_params_down_week.pkl')[1] # first num is estim of A, second of tau
##

# load choices' bins' boundaries
quality_bins = pd.read_pickle(out_dir + 'choice_qualityansw_bins_week.pkl')
quantity_bins = pd.read_pickle(out_dir + 'choice_quantityansw_bins_week.pkl')

# load choice values
quality_choices = pd.read_pickle(out_dir + 'choice_qualityansw_week.pkl')
leftbound_quality = [0] + [i.left for i in quality_choices.index] + [quality_bins.iloc[-1]]
quantity_choices = pd.read_pickle(out_dir + 'choice_quantityansw_week.pkl')
leftbound_quantity = [0] + [i.left for i in quantity_choices.index] + [quantity_bins.iloc[-1]]

# bin quality variable, creating the choice variable
hist.loc[:,'quality'] = pd.cut(hist['quality_orig'], bins=leftbound_quality, labels=[0] + quality_choices.tolist(), right=True, include_lowest=True)
hist.loc[:,'quality'] = hist['quality'].astype(float)
# bin quantity variable
hist.loc[:,'numAnswers'] = pd.cut(hist['numAnswers_orig'], bins=leftbound_quantity, labels=[0] + quantity_choices.tolist(), right=True, include_lowest=True)
hist.loc[:,'numAnswers'] = hist['numAnswers'].astype(float)

# expected number of edits received on new answers given 1) previous period realization of states, 2) choices
hist.loc[:,'numedits_totalothers_accepted'] = EAE.predict(hist[['quality','AnswerNum','Seniority_days']])   

# expected number of upvotes/downvotes received on new answers given 1) previous period realization of states, 2) choices
# upvotes
hist.loc[hist['numAnswers']>0,'EUV'] = hist['numAnswers'] * EUV.predict(hist[['quality','AnswerNum','Seniority_days','numedits_totalothers_accepted']])
hist.loc[hist['numAnswers']==0,'EUV'] = 0
# downvotes
hist.loc[hist['numAnswers']>0,'EDV'] = hist['numAnswers'] * EDV.predict(hist[['quality','AnswerNum','Seniority_days','numedits_totalothers_accepted']])
hist.loc[hist['numAnswers']==0,'EDV'] = 0

def expo(t, A, tau): # exponential decay function; A = aplitude, tau = decay time
	return A * np.exp(-t/tau) 

def lambda_f(x, tau):
    '''
    x  := vector of per period values of A
    ''' 
    time_matrix = np.tile(np.arange(0,len(x)),(len(x),1))
    time_matrix = np.maximum(time_matrix - np.arange(0,len(x))[:,np.newaxis], np.zeros((len(x),len(x))))
    means = expo(time_matrix,x.values[:,np.newaxis],tau)
    means = np.triu(means,1)
    means = np.sum(means,axis=0)
    return means

# make sure periods are sorted
hist.sort_values(by=['user','periods'], inplace=True)

# expected number of upvotes arriving from past actions
hist.loc[:,'lambda_up'] = hist.groupby('user')['EUV'].transform(lambda x: lambda_f(x,tau_up))

# expected number of downvotes arriving from past actions
hist.loc[:,'lambda_down'] = hist.groupby('user')['EDV'].transform(lambda x: lambda_f(x,tau_down))

#################################################################################################################
### choice of number of edits
#################################################################################################################
print('start edits')
# fix edit choice variable
edits_bins = pd.read_pickle(out_dir + 'choice_quantityedits_bins_week.pkl')
edits_choices = pd.read_pickle(out_dir + 'choice_quantityedits_week.pkl')
leftbound_edits = [0] + [i.left for i in edits_choices.index] + [edits_bins.iloc[-1]]
hist.loc[:,'numEdits'] = pd.cut(hist['numEdits_orig'], bins=leftbound_edits, labels=[0] + edits_choices.tolist(), right=True, include_lowest=True)
hist.loc[:,'numEdits'] = hist['numEdits'].astype(float)

# tot edits received
hist['EO'] = hist['EOd'] + hist['EOs']

# tot edits received implemented
hist['EOimpl'] = hist['EOd'] + hist['EOsa']

### save small dataset for estimation
# public beta day of site 
if qa_name == 'apple/':
    pb = pd.Timestamp(2010,8,24)
elif qa_name == 'ell/':
    pb = pd.Timestamp(2013,1,30)
# drop obs before public beta (still experience vars ecc don't neglect it, i just don't want to use it for inference on choices)
hist = hist.loc[hist['day']>=pb]

hist.sort_values(by=['user','day'], inplace=True)
# lag state variables

# experience: AnswerNum (for Seniority no shift because on first day of registration Seniority==0)
hist.loc[:,'AnswerNumL'] = hist.groupby('user')['AnswerNum'].transform(lambda x: x.shift(1,fill_value=0))
# rep cum
hist.loc[:,'rep_cumL'] = hist.groupby('user')['rep_cum'].transform(lambda x: x.shift(1,fill_value=0))
# reciprocity
hist.loc[:,'EOL'] = hist.groupby('user')['EO'].transform(lambda x: x.shift(1,fill_value=0))
hist.loc[:,'EOimplL'] =  hist.groupby('user')['EOimpl'].transform(lambda x: x.shift(1,fill_value=0))
hist.loc[:,'received_answersL'] =  hist.groupby('user')['received_answers'].transform(lambda x: x.shift(1,fill_value=0))
hist.loc[:,'received_1st_answersL'] =  hist.groupby('user')['received_1st_answers'].transform(lambda x: x.shift(1,fill_value=0))
# altruism
hist.loc[:,'numAcceptedAnswersL'] = hist.groupby('user')['numAcceptedAnswers'].transform(lambda x: x.shift(1,fill_value=0))
hist.loc[:,'EditedPostIsAcceptedL'] = hist.groupby('user')['EditedPostIsAccepted'].transform(lambda x: x.shift(1,fill_value=0))
hist.loc[:,'EditedPostsUpVotesL'] = hist.groupby('user')['EditedPostsUpVotes'].transform(lambda x: x.shift(1,fill_value=0))
hist.loc[:,'EditedPostsDownvotesL'] = hist.groupby('user')['EditedPostsDownvotes'].transform(lambda x: x.shift(1,fill_value=0))
# # rank
# hist.loc[:,'rankL'] = hist.groupby('user')['rank'].transform(lambda x: x.shift(1,fill_value=0))
# hist.loc[:,'topPercentL'] = hist.groupby('user')['topPercent'].transform(lambda x: x.shift(1,fill_value=0))
# questions made
hist.loc[:,'numQuestionsL'] = hist.groupby('user')['numQuestions'].transform(lambda x: x.shift(1,fill_value=0))
# thresholds reached
hist.loc[:,'TcumL'] = hist.groupby('user')['Tcum'].transform(lambda x: x.shift(1,fill_value=0))
#hist.loc[:,'ReachedTL'] = hist.groupby('user')['ReachedT'].transform(lambda x: x.shift(1,fill_value=0))

# # change reciprocity abd altruism variable to cumulative
# hist.loc[:,'received_answers_cum'] = hist.groupby('user')['received_answers'].cumsum()
# hist.loc[:,'EOimpl_cum'] = hist.groupby('user')['EOimpl'].cumsum()
# hist.loc[:,'numAcceptedAnswers_cum'] = hist.groupby('user')['numAcceptedAnswers'].cumsum()
# hist.loc[:,'EditedPostsUpVotes_cum'] = hist.groupby('user')['EditedPostsUpVotes'].cumsum()

# # scale EditedPostsUpVotes_cum and numAcceptedAnswers_cum
# hist.loc[:,'EditedPostsUpVotes_cum'] = np.maximum(0,np.log(hist['EditedPostsUpVotes_cum']))
# hist.loc[:,'numAcceptedAnswers_cum'] = np.maximum(0,np.log(hist['numAcceptedAnswers_cum']))

# hist.loc[:,'numQuestions_cum'] = hist.groupby('user')['numQuestions'].cumsum()

essential_cols = ['day','periods', 'user', 'rep_cumL', 'AnswerNumL', 'Seniority_days',
       'EOimplL', 'EOL', 'received_answersL','numQuestionsL','numAcceptedAnswersL',
       'EditedPostIsAcceptedL', 'EditedPostsUpVotesL', 'EditedPostsDownvotesL','TcumL',
       'avail', 'quality', 'numAnswers', 'lambda_up', 'lambda_down', 'numEdits']
       #'ReachedTL','rankL','topPercentL'
       

hist[essential_cols].to_csv(out_dir + 'UserHist_Csmall_week.csv', index = False)

### add choicenum and other relevant variables ###
hist = pd.read_csv(out_dir + 'UserHist_Csmall_week.csv', parse_dates=['day'])

# day num
if not 'date2num_week.pkl' in os.listdir(out_dir):
    dates = pd.date_range(start=hist['day'].min(), end=hist['day'].max(), freq='1W')
    dates2num = pd.DataFrame({'day':dates,'datenum':np.arange(len(dates))})
    pd.to_pickle(dates2num, out_dir + 'date2num_week.pkl')
else:
    dates2num = pd.read_pickle(out_dir + 'date2num_week.pkl')
    
hist = pd.merge(hist, dates2num, on='day', validate='m:1')

# choice
hist.loc[:,'choice'] = hist.apply(lambda row: (row['numAnswers'],row['quality'],row['numEdits']), axis=1)

choices = hist['choice'].unique()
choices = np.sort(choices)

# choice to num dict
if not 'choice2num_week.pkl' in os.listdir(out_dir) or not 'num2choice_week.pkl' in os.listdir(out_dir): 
    choice_num2tupl = {j:i for i, j in list(zip(choices, np.arange(len(choices))))}
    choice_tupl2num = {i:j for i, j in list(zip(choices, np.arange(len(choices))))}
    
    pd.to_pickle(choice_num2tupl, out_dir + 'num2choice_week.pkl')
    pd.to_pickle(choice_tupl2num, out_dir + 'choice2num_week.pkl')
else:
    choice_num2tupl = pd.read_pickle(out_dir + 'num2choice_week.pkl')
    choice_tupl2num = pd.read_pickle(out_dir + 'choice2num_week.pkl')

hist.loc[:,'choicenum'] = hist['choice'].apply(lambda x: choice_tupl2num[x])

# types
types = pd.read_csv(out_dir + 'individual_chars_dummies_wgroups.csv',usecols=['Id','user_types'])
types.rename(columns={'Id':'user'}, inplace=True)

hist = pd.merge(hist, types, on='user', how='left', validate='m:1')

# save 
hist = hist.sort_values(by=['user','periods'])
essential_cols = ['day','periods', 'user', 'rep_cumL', 'AnswerNumL', 'Seniority_days',
       'EOimplL', 'EOL', 'received_answersL','numQuestionsL','numAcceptedAnswersL',
       'EditedPostIsAcceptedL', 'EditedPostsUpVotesL', 'EditedPostsDownvotesL','TcumL',
       'avail', 'quality', 'numAnswers', 'lambda_up', 'lambda_down', 'numEdits','datenum',
       'choicenum','user_types']
       #'ReachedTL','rankL','topPercentL'
hist[essential_cols].to_csv(out_dir + 'UserHist_Csmall_week.csv', index = False)

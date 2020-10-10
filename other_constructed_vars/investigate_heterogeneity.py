#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 14:12:07 2020

@author: jacopo
"""

'''
Check heterogeneity in behavior
'''

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, FastICA
import matplotlib.pyplot as plt
import seaborn as sns
import classes as cl
import statsmodels.api as sm
import statsmodels.formula.api as smf


directory_data = '/home/jacopo/OneDrive/Dati_Jac_locali/stack/'
#qa_name = 'apple/'
qa_name = 'ell/'

dta = pd.read_csv(directory_data + qa_name + 'individual_chars.csv')

# transform data so to have only dummies (year of registration not used because not really a stretegic deicision)
median = dta.loc[dta['lenAboutMeALL']>0,'lenAboutMeALL'].median()
dta.loc[:,'sizeAboutMe'] = pd.cut(dta['lenAboutMeALL'], bins=[-1,0,median,dta['lenAboutMeALL'].max()], labels=False)
#dm = pd.get_dummies(dta['sizeAboutMe'], prefix='sizeAboutMe')

dta.loc[:,'sizelinksAboutMe'] = pd.cut(dta['numLinksAboutMe'], bins=[-1,0,3,dta['numLinksAboutMe'].max()], labels=False)
#dm2 = pd.get_dummies(dta['sizelinksAboutMe'], prefix='sizelinksAboutMe')

#dm3 = pd.get_dummies(dta['yearRegistration'], prefix='year')

dta[['Id','has_fullname','has_website', 'has_location', 
     'has_linkedin','sizeAboutMe','sizelinksAboutMe',
     'yearRegistration']].to_csv(directory_data + qa_name + 'individual_chars_dummies.csv', index=False)

'''
ANALYSIS IN R --> individual_heterogeneity.R
'''
# import output from R

dta = pd.read_csv(directory_data + qa_name + 'individual_chars_dummies_wgroups.csv', index_col=0)
'''
# save for stata
dtastata = dta[['Id','user_types']]
dtastata.rename(columns={'Id':'user'}, inplace=True)
dtastata.loc[:,'user'] = dtastata['user'].astype(str)
dtastata.to_stata(directory_data + qa_name + 'individual_wgroups.dta', write_index=False)
'''
# badges correlated with type
badges = pd.read_csv(directory_data + qa_name + 'badge_hist.csv')
badges = badges.groupby('UserId').sum()
badges.reset_index(inplace=True)

badges = pd.merge(badges, dta[['Id','user_types']], left_on='UserId', right_on='Id', how='inner', validate='1:1')

barplotdt = badges.groupby('user_types')[['Gold','Silver','Bronze']].agg(['mean','std','sem','count'])

bar_width = 0.2
fig, ax = plt.subplots()
plt.grid(axis='y')
bronze = ax.bar(barplotdt.index.values - bar_width, barplotdt[('Bronze','mean')].values,
                width=bar_width, yerr=barplotdt[('Bronze','sem')], label='Bronze badges')
silver = ax.bar(barplotdt.index.values, barplotdt[('Silver','mean')].values,
                width=bar_width, yerr=barplotdt[('Silver','sem')], label='Silver badges')
gold = ax.bar(barplotdt.index.values + bar_width, barplotdt[('Gold','mean')].values,
                width=bar_width, yerr=barplotdt[('Gold','sem')], label='Gold badges')
plt.xticks(barplotdt.index.values, labels=['Type 1','Type 2','Type 3'])
plt.legend()
plt.title('Average number of badges obtained by each user')
plt.xlabel('User type')
plt.tight_layout()
# saved numbadgesXtype.png

### other correlations with type
user = cl.Users(qa_name, directory_data, out_type='df').users()
user.loc[:,'CreationDate'] = user['CreationDate'].apply(cl.date)
designdate = pd.Timestamp(2016,2,25)
user_beforeGrad = user.loc[user['CreationDate']<designdate,'Id']
user_beforeGrad = user_beforeGrad.astype(int)

hist = pd.read_csv(directory_data + qa_name + 'UserHist.csv', index_col=0, parse_dates=True)
hist.index.name = 'day'
hist.reset_index(inplace=True)

hist = pd.merge(hist, dta[['Id','user_types']], left_on='user', right_on='Id', how='inner', validate='m:1')
# tot quality
hist['totquality'] = hist['numAnswers'] * hist['quality']
# is editor
hist.loc[(hist['day']<designdate) & (hist['rep_cum']>=1000),'isEditor'] = 1
hist.loc[(hist['day']>=designdate) & (hist['rep_cum']>=2000),'isEditor'] = 1
hist.loc[:,'isEditor'].fillna(0, inplace=True)

## TIME TO BECOME EDITOR
# early users history before graduation (those ones looking at the 1000 points threhsolds AND before they get to the design date)
EUhist = hist.loc[(hist['user'].isin(user_beforeGrad)) & ((hist['day']<designdate))]
reached = EUhist.groupby('user')['isEditor'].max()
reached = reached[reached==1].index.tolist()

gotit = EUhist.loc[EUhist['isEditor']==1].groupby('user')['Seniority_days'].first().reset_index()
gotit['reached'] = 1
notgotit = EUhist.loc[~EUhist['user'].isin(reached)].groupby('user')['Seniority_days'].last().reset_index()
notgotit['reached'] = 0

dfEU = pd.concat([gotit, notgotit], axis=0)
dfEU = pd.merge(dfEU, dta[['Id','user_types']], left_on='user', right_on='Id', how='inner', validate='1:1')
dfEU = pd.concat([dfEU, pd.get_dummies(dfEU['user_types'], prefix='type')], axis=1)

mod = smf.phreg("Seniority_days ~ 0 + type_2 + type_3", status=dfEU['reached'].values,
                data=dfEU, ties="efron")
rslt = mod.fit()
print(rslt.summary())

fig, ax = plt.subplots()
for tp in range(1,4):
    sf = sm.SurvfuncRight(dfEU.loc[dfEU['user_types']==tp,"Seniority_days"], dfEU.loc[dfEU['user_types']==tp,"reached"])
    sf.plot(ax)
li = ax.get_lines()
li[1].set_visible(False) # removes crosses (not clear what they mean)
li[3].set_visible(False)
li[5].set_visible(False)
plt.legend((li[0], li[2], li[4]), ('Type 1', 'Type 2', 'Type 3'))
plt.ylim(0.8, 1)
plt.ylabel("Proportion not editor")
plt.xlabel("Days of participation")
plt.title("Survival function - time to become Editor (before website' design)")
plt.tight_layout()
# saved as survivalByTypes_earlyT.png

# late users history (those ones never concerned by the change in threshold)
LUhist = hist.loc[(~hist['user'].isin(user_beforeGrad))]
reached = LUhist.groupby('user')['isEditor'].max()
reached = reached[reached==1].index.tolist()

gotit = LUhist.loc[LUhist['isEditor']==1].groupby('user')['Seniority_days'].first().reset_index()
gotit['reached'] = 1
notgotit = LUhist.loc[~LUhist['user'].isin(reached)].groupby('user')['Seniority_days'].last().reset_index()
notgotit['reached'] = 0

dfLU = pd.concat([gotit, notgotit], axis=0)
dfLU = pd.merge(dfLU, dta[['Id','user_types']], left_on='user', right_on='Id', how='inner', validate='1:1')
dfLU = pd.concat([dfLU, pd.get_dummies(dfLU['user_types'], prefix='type')], axis=1)

mod = smf.phreg("Seniority_days ~ 0 + type_2 + type_3", status=dfLU['reached'].values,
                data=dfLU, ties="efron")
rslt = mod.fit()
print(rslt.summary())

fig, ax = plt.subplots()
for tp in range(1,4):
    sf = sm.SurvfuncRight(dfLU.loc[dfLU['user_types']==tp,"Seniority_days"], dfLU.loc[dfLU['user_types']==tp,"reached"])
    sf.plot(ax)
li = ax.get_lines()
li[1].set_visible(False) # removes crosses (not clear what they mean)
li[3].set_visible(False)
li[5].set_visible(False)
plt.legend((li[0], li[2], li[4]), ('Type 1', 'Type 2', 'Type 3'))
plt.ylim(0.9, 1)
plt.ylabel("Proportion not editor")
plt.xlabel("Days of participation")
plt.title("Survival function - time to become Editor (after website' design)")
plt.tight_layout()

fig, ax = plt.subplots()
for tp in range(1,4):
    sf = sm.SurvfuncRight(dfLU.loc[dfLU['user_types']==tp,"Seniority_days"], dfLU.loc[dfLU['user_types']==tp,"reached"])
    sf.plot(ax)
li = ax.get_lines()
li[1].set_visible(False) # removes crosses (not clear what they mean)
li[3].set_visible(False)
li[5].set_visible(False)
plt.legend((li[0], li[2], li[4]), ('Type 1', 'Type 2', 'Type 3'))
plt.ylim(0.9, 1)
plt.ylabel("Proportion not editor")
plt.xlabel("Days of participation")
plt.title("Survival function - time to become Editor (after website' design)")
plt.tight_layout()
# saved as survivalByTypes_lateT.png


### who is elected
elections = pd.read_csv(directory_data + qa_name + 'elections.csv', parse_dates=[0,1,2])
candidates = elections.groupby('user')['iswinner'].max()
candidates = candidates.reset_index()

groups = dta[['Id','user_types']]
groups = pd.merge(groups, candidates, left_on='Id', right_on='user', how='right', validate='1:1')
numwl = groups.groupby('user_types')['iswinner'].agg(['count','sum'])

bar_width = 0.3
fig, ax = plt.subplots()
plt.grid(axis='y')
candidate = ax.bar(numwl.index.values - bar_width/2, numwl['count'].values,
                width=bar_width, label='num. Candidates')
winners = ax.bar(numwl.index.values + bar_width/2, numwl['sum'].values,
                width=bar_width, label='num. Elected')
plt.xticks(numwl.index.values, labels=['Type 1','Type 2','Type 3'])
plt.legend()
plt.title("Elections' participants")
plt.xlabel('User type')
plt.tight_layout()
# saved as elections_participants.png
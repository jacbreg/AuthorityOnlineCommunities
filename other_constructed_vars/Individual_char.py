#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 19:34:36 2019

@author: jacopo
"""

''' script to extract individual char from user profiles'''

import pandas as pd
import classes as cl
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import numpy as np
stopwords = stopwords.words('english')


directory_data = '/home/jacopo/OneDrive/Dati_Jac_locali/stack/'
#qa_name = 'apple/'
qa_name = 'ell/'

usersdf = cl.Users(qa_name, directory_data, out_type='df').users()

def numwords_aboutme(x):
    if type(x)==str:
        txt = x.lower().split()
        return len([i for i in txt if not i in stopwords])
    else:
        return 0
    
def numwords_aboutmeALL(x):
    if type(x)==str:
        txt = x.lower().split()
        return len(txt)
    else:
        return 0

def has_fullname(x):
    if re.search('[A-Z][a-z]+\s[A-Z][a-z]', x):
        return 1
    else:
        return 0

def numlinks_aboutme(x):
    if type(x)==str:
        soup = BeautifulSoup(x)
        return len(soup.find_all('a'))
    else:
        return 0

def has_linkedin(row):
    if type(row['WebsiteUrl'])==str and re.search('linkedin',row['WebsiteUrl']):
        return 1
    elif type(row['AboutMe'])==str and re.search('linkedin',row['AboutMe']):
        return 1
    else:
        return 0

usersdf.loc[:,'lenAboutMe'] = usersdf['AboutMe'].apply(numwords_aboutme)
usersdf.loc[:,'lenAboutMeALL'] = usersdf['AboutMe'].apply(numwords_aboutmeALL)
usersdf.loc[:,'has_fullname'] = usersdf['DisplayName'].apply(has_fullname)
usersdf.loc[:,'numLinksAboutMe'] = usersdf['AboutMe'].apply(numlinks_aboutme)

usersdf.loc[usersdf['WebsiteUrl'].notna(),'has_website'] = 1
usersdf.loc[usersdf['WebsiteUrl'].isna(),'has_website'] = 0

usersdf.loc[usersdf['Location'].notna(),'has_location'] = 1
usersdf.loc[usersdf['Location'].isna(),'has_location'] = 0

usersdf.loc[:,'has_linkedin'] = usersdf.apply(has_linkedin, axis=1)

usersdf.loc[:,'CreationDate'] = usersdf['CreationDate'].apply(cl.date)
usersdf.loc[:,'yearRegistration'] = usersdf['CreationDate'].apply(lambda x: x.year)

# age no more available in data dump
# usersdf.loc[usersdf['Age'].notna(),'has_age'] = 1
# usersdf.loc[usersdf['Age'].isna(),'has_age'] = 0
 
cols = ['Id','lenAboutMe','lenAboutMeALL','has_fullname','numLinksAboutMe','has_website',
        'has_location','has_linkedin','yearRegistration'] #,'has_age']
usersdf[cols].to_csv(directory_data + qa_name + 'individual_chars.csv', index=False)

# yes/no table
usersdf['has_bio'] = np.where(usersdf['lenAboutMeALL']>0,1,0)
usersdf['has_links'] = np.where(usersdf['numLinksAboutMe']>0,1,0)

share = usersdf[['has_fullname','has_website','has_location','has_linkedin','has_bio','has_links']].mean()
share = share * 100
share.name = 'Share of users'
print(share.round(2).to_latex())

# lenght variables
pos_bio = usersdf.loc[usersdf['has_bio']==1,['lenAboutMe','lenAboutMeALL']].describe().round(2)
pos_links = usersdf.loc[usersdf['has_links']==1,['numLinksAboutMe']].describe().round(2)
print(pd.concat([pos_bio,pos_links], axis=1).to_latex())

### BY TYPE
types = pd.read_csv(directory_data + qa_name + 'individual_chars_dummies_wgroups.csv',usecols=['Id','user_types'], dtype={'Id':'str'})
usersdf = pd.merge(usersdf, types, on='Id', how='inner',validate='1:1')
# yes/no table
share = usersdf.groupby('user_types')[['has_fullname','has_website','has_location','has_linkedin','has_bio','has_links']].mean()
share = share * 100
print(share.round(2).to_latex())
# lenght variables
pos_bio = usersdf.loc[usersdf['has_bio']==1,['user_types','lenAboutMe','lenAboutMeALL']].groupby('user_types').describe().stack().round(2)
pos_links = usersdf.loc[usersdf['has_links']==1,['user_types','numLinksAboutMe']].groupby('user_types').describe().stack().round(2)
print(pd.concat([pos_bio,pos_links], axis=1).to_latex())

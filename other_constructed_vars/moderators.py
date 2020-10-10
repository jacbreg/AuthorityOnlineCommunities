#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 08:55:26 2018

@author: jacopo
"""

'''
list of moderators
'''
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import numpy as np

directory_data = '/home/jacopo/OneDrive/Dati_Jac_locali/stack/'


### ask different / apple 
qa_name = 'apple/'

headers = {
    'User-Agent': 'jacrobot',
    'From': 'jak117@hotmail.it' 
    }

url = 'https://apple.stackexchange.com/election'
r = requests.get(url, headers=headers)
content = r.content
soup = BeautifulSoup(content, 'html.parser')


election_pages = soup.find_all('a', {'href': re.compile('election/[0-9]')})

alldta = []
for page in election_pages:
    nr = requests.get('https://apple.stackexchange.com' + page['href'])
    ncontent = nr.content
    nsoup = BeautifulSoup(ncontent, 'html.parser')
    
    year = re.findall('[0-9]{4}',nsoup.find('div', {'class':'subheader'}).h1.string)[0]

    keyssidebar = nsoup.find('div', {'id':'sidebar'}).div.find_all('p',{'class':'label-key'})
    keyssidebar = ['nomination period began','election began','election ended','moderator candidates','moderator positions available']
    valuesssidebar = nsoup.find('div', {'id':'sidebar'}).div.find_all('p',{'class':'label-value'})

    candidates = nsoup.find_all('a',{'href':re.compile('^/users/[0-9]+'), 'class':False}) # first three are winners
    winners = [re.findall('[0-9]+',i['href'])[0] for i in candidates[:3]]# ALL ELECTIONS HAVE MAX 3 WINNERS
    candidates = list(set([re.findall('/users/([0-9]+)',i['href'])[0] for i in candidates]))
    
    obs = {}
    for counter, value in enumerate(keyssidebar):
        if 'title' in valuesssidebar[counter].attrs.keys():
            obs[value] = np.repeat(np.array([valuesssidebar[counter]['title']]), len(candidates))
        else:
            obs[value] = np.repeat(np.array([valuesssidebar[counter].string]), len(candidates))
            
    pagedta = pd.DataFrame(obs)
    pagedta.loc[:,'user'] = np.array(candidates)
    pagedta.loc[:,'iswinner'] = pagedta['user'].apply(lambda x: 1 if x in winners else 0)
    pagedta.loc[:,'year'] = year
    
    alldta.append(pagedta)

alldta = pd.concat(alldta)

alldta.to_csv(directory_data + qa_name + 'elections.csv', index=False)

# moderators may have resigned some time after their appointment or removed (very rare)
# otherwise, moderators are elected for life

### english language learners
qa_name = 'ell/'

headers = {
    'User-Agent': 'jacrobot',
    'From': 'jak117@hotmail.it' 
    }

url = 'https://ell.stackexchange.com/election'
r = requests.get(url, headers=headers)
content = r.content
soup = BeautifulSoup(content, 'html.parser')


election_pages = soup.find_all('a', {'href': re.compile('election/[0-9]')})

alldta = []
for page in election_pages:
    nr = requests.get('https://ell.stackexchange.com' + page['href'])
    ncontent = nr.content
    nsoup = BeautifulSoup(ncontent, 'html.parser')
    
    year = re.findall('[0-9]{4}',nsoup.find('div', {'class':'subheader'}).h1.string)[0]

    keyssidebar = nsoup.find('div', {'id':'sidebar'}).div.find_all('p',{'class':'label-key'})
    keyssidebar = ['nomination period began','election began','election ended','moderator candidates','moderator positions available']
    valuesssidebar = nsoup.find('div', {'id':'sidebar'}).div.find_all('p',{'class':'label-value'})

    candidates = nsoup.find_all('a',{'href':re.compile('^/users/[0-9]+'), 'class':False}) # first three are winners
    winners = [re.findall('[0-9]+',i['href'])[0] for i in candidates[:3]]# ALL ELECTIONS HAVE MAX 3 WINNERS
    candidates = list(set([re.findall('/users/([0-9]+)',i['href'])[0] for i in candidates]))
    
    obs = {}
    for counter, value in enumerate(keyssidebar):
        if 'title' in valuesssidebar[counter].attrs.keys():
            obs[value] = np.repeat(np.array([valuesssidebar[counter]['title']]), len(candidates))
        else:
            obs[value] = np.repeat(np.array([valuesssidebar[counter].string]), len(candidates))
            
    pagedta = pd.DataFrame(obs)
    pagedta.loc[:,'user'] = np.array(candidates)
    pagedta.loc[:,'iswinner'] = pagedta['user'].apply(lambda x: 1 if x in winners else 0)
    pagedta.loc[:,'year'] = year
    
    alldta.append(pagedta)

alldta = pd.concat(alldta)

alldta.to_csv(directory_data + qa_name + 'elections.csv', index=False)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 17:28:04 2020

@author: jacopo
"""
'''
scrape rep points of ell users
'''

import requests
from bs4 import BeautifulSoup
import time
import re
import pandas as pd
import classes as cl
import os

directory_data = '/home/jacopo/OneDrive/Dati_Jac_locali/stack/'
out_dir = '/home/jacopo/OneDrive/Dati_Jac_locali/stack/ell/'

users = cl.Answers('ell/', directory_data, out_type='df').answers()
users = users.loc[users['']]
users = users['OwnerUserId'].unique().tolist() # this exclude people with 100 points as joining bonus but no answers

users2 = cl.Users('ell/', directory_data, out_type='df').users()
users2.loc[:,'Reputation'] = users2['Reputation'].astype(int)
users2 = users2.loc[users2['Reputation']>1]
users2 = users2['Id'].tolist() # this exclude people with action but who either didn't get any point or got only negative points (rep cannot go minus zero so they are not counted)

users_touse = [i for i in users if i in users2]


done = []
notfound = []

already_saved=pd.read_csv(out_dir + 'ELLrep.csv', dtype={'user':str})
done.extend(already_saved['user'].unique().tolist())

count = 0
for user in users_touse:
    
    if user in done:
        count += 1
        continue
    
    print(len(users)-count, 'missing')
    
    url = 'https://ell.stackexchange.com/users/{}/?tab=reputation&sort=graph'.format(user)
    
    headers = {
        'User-Agent': 'jacrobot',
        'From': 'jak117@hotmail.it' 
        }
    
    r = requests.get(url, headers=headers)
    content = r.content
    soup = BeautifulSoup(content, 'html.parser')
    
    data = soup.find_all('script')
    is_found = False
    for script in data:
        if re.search('rawData = (.+);', str(script)):
            rawdata = re.findall('rawData = (.+);', str(script))
            is_found = True
            break
    
    if is_found==False:
        print('Error: data not found of user {}'.format(user))
        notfound.append(user)
        count +=1
        continue
        
    data = eval(rawdata[0])
    
    df = pd.DataFrame(data, columns=['year','month','day','rep'])
    df.loc[:,'date'] = df.apply(lambda x: pd.Timestamp(x['year'],x['month'],x['day']), axis=1)
    
    df.loc[:,'user'] = user
    df = df[['date','user','rep']]
    # keep only non-zero values
    df = df.loc[df['rep']!=0]
    
    if 'ELLrep.csv' in os.listdir(out_dir):
        df.to_csv(out_dir + 'ELLrep.csv', mode='a', header=False, index=False) 
    else:
        df.to_csv(out_dir + 'ELLrep.csv', mode='a', index=False)    
    
    done.append(user)
    time.sleep(2)
    count += 1



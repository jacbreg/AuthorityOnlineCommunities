#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 23:54:54 2020

@author: jacopo
"""
'''
 LIST OF PRIVILEGES WITH DESCRIPTION  
'''
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re

qa_name = 'ell/'

directory = '/home/jacopo/OneDrive/Dati_Jac_locali/stack/'
out_dir = directory + qa_name

d = 'https://ell.stackexchange.com/help/privileges'

rbase = requests.get(d)
page = rbase.text
soup = BeautifulSoup(page, 'lxml')
table = soup.find_all('div', {'id' : 'privilege-table'})
privileges = table[0].find_all('a')

privilegeslist = []
for priv in privileges:
    short = priv.find_all('div', {'class':"short-description grid ai-center"})[0].text #.find_all('div',{'class':"grid--cell"}).text
    short = re.findall('[a-z][a-z\s]+[a-z]',short)[0]
    long = priv.find_all('div', {'class':"long-description"})[0].text
    rep = priv.find_all('div', {'class':"rep-level"})[0].text
    rep = re.sub(',','',rep)
    short = re.sub(',','-',short)
    short = re.sub(';','-',short)
    long = re.sub(',','-',long)
    long = re.sub(';','-',long)
    infodict = {'rep-level': int(rep), 'short-description': short, 'long-description': long}
    privilegeslist.append(infodict)
    
privilegesdf = pd.DataFrame(privilegeslist, columns=['rep-level', 'short-description', 'long-description'])
print(privilegesdf.to_latex())

privilegesdf.to_csv(out_dir + 'thresholds.csv', index=False)

print(priv[['short-description','type','rep-level','rep-level-publicbeta']].to_latex(index=False, na_rep=""))


x = '''<pre>

Privilege                                 Priv β Public β  Designed

───────────────────────────────────────── ────── ──────── ─────────

Participate in meta                            5        5       5†

Skip lecture on how to ask                     -        -      10†

Create community-wiki answers                 10       10      10

Remove new-user restrictions¹                  1       10      10

Vote up                                        1       15      15

Flag posts                                    15       15      15

Post self-answered questions                  15       15      15

Comment everywhere                             1       50      50†

Set bounties                                  75       75      75

Edit community wikis                           1      100     100

Vote down                                      1      125     125†

Create tags                                    1      150     300†

Vote in moderator elections                    -      150     150&diams; <---------- added manually

Association bonus²                           200      200     200        <---------- added manually

Shown in network reputation graph and flair  200      200     200        <---------- added manually

Shown as "beta user" on Area 51              200      200       -

Reduced advertisements                         -        -     200

Reputation leagues, top x% link in profile   201      201     201&diams; <---------- added manually

Qualify for first Yearling badge             201      201     201&diams; <---------- added manually

View close votes³                              1      250     250

Run for moderator                              -      300     300†       <---------- added manually

Access review queues⁴                        350      350     500†

See vote counts                              100      750    1000        <---------- added manually

Edit freely, SE and LQP queue⁵               500     1000    2000

No popup asking to comment when downvoting  2000     2000    2000&diams;

Non-nofollow link in user profile⁶          2000     2000    2000

Suggest tag synonyms                        1250     1250    2500

Vote to close and reopen                      15      500    3000

Review tag wiki edits                        750     1500    5000

Moderator tools⁷                            1000     2000   10000

Reduce captchas                             1000?    2000   10000

Protect questions                           1750     3500   15000

Trusted user⁸                               2000     4000   20000

Access to site analytics                    2500     5000   25000

</pre>'''

x = markdown(x)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue July 14 

@author: jacopo
"""

'''
Beliefs on the evolution of the variable 'availability' - version at week level
'''

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

#qa_name = 'apple/'
qa_name = 'ell/'
directory = '/home/jacopo/OneDrive/Dati_Jac_locali/stack/' + qa_name

availability = pd.read_csv(directory + 'availability.csv',index_col=0, parse_dates=True )

# complete index
compl_avail_index = pd.date_range(start=availability.index.min(), end=availability.index.max(), freq='D')
availability = availability.reindex(compl_avail_index, method='ffill')

availability = availability.resample('W')['numQ'].max()
# remove first and last weeks that may be less than 7 days
availability = availability.iloc[1:-1]
availability = pd.DataFrame(availability)

availability.loc[:,'t'] = np.arange(0,len(availability),step=1)
availability.loc[:,'const'] = 1

# time trend of avail
modfit = sm.OLS(availability['numQ'].values,availability[['const','t']].values).fit()
print(modfit.summary())

# public beta day of site 
if qa_name == 'apple/':
    pb = pd.Timestamp(2010,8,24)
elif qa_name == 'ell/':
    pb = pd.Timestamp(2013,1,30)

# time trend of avail ONLY AFTER PUBLIC BETA
dta = availability.loc[availability.index>=pb]
modfit2 = sm.OLS(dta['numQ'].values,dta[['const','t']].values).fit()
print(modfit2.summary())
rate_avail = modfit2.params[1]
pd.to_pickle(rate_avail, directory + 'rate_avail_week.pkl')

# time trend of avail ONLY BEFORE PUBLIC BETA
dtabf = availability.loc[availability.index<pb]
modfit3 = sm.OLS(dtabf['numQ'],dtabf[['const','t']]).fit()
print(modfit3.summary()) # basically no increase (in estimation set as coef=0)

def trend(x, const, coef):
    return const + coef*x


plt.figure()
plt.plot(availability.index,availability['numQ'])
ymin, ymax = plt.ylim()
plt.plot(dta.index, trend(dta['t'], modfit2.params[0], modfit2.params[1]), linestyle='dashdot', color='black', label='Linear Trend')
plt.axvline(x=pb, color='darkred', linestyle='--')
plt.ylabel('Num. questions without accepted answer')
plt.xlabel('Days from website creatiion')
plt.text(pb, 20000,'Public Beta starts', rotation='90')
plt.legend()
# saved as avail_evolution.png
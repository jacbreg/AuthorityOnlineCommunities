#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 11:39:40 2020

@author: jacopo
"""

'''
Estimation for true data aggregated at weekly period
Model:
    U = cost_answering + cost_editing + rep_cum + isEditor (+ interactions isEditor with costs)
cost_answering = quality + quantity^(scarsity)
scarsity is in [1,inf) , since scarsity = 1 / ((avail- min(avail))/(max(avail)-min(avail)))
cost_editing = numedits (for now linear)

Beliefs:
    - evolution of points: 
        x_up_t ~ P(lambda_up_t + EUV_t)
        x_down_t ~ P(lambda_down_t + EDV_t)
        rep_cum_t+1 = rep_cum_t + 10*(x_up_t) - 2*(x_down_t)
    - evolution of experience: deterministic 
        AnswerNum_t+1 = AnswerNum_t + quantity_t
        Seniority_days_t+1 = Seniority_days_t
    - evolution of avail:
        people estiamte non-topic-specific time trend of evolution of avail:  around 15 per period (saved coef) 
        
'''
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import scipy.optimize as sopt
import math
import re
import os

import FunctExpStates as FES

#qa_name = 'apple/'
qa_name = 'ell/'

directory = '/home/jacopo/OneDrive/Dati_Jac_locali/stack/' + qa_name
out_dir = '/home/jacopo/OneDrive/Dati_Jac_locali/stack/'+ qa_name + 'DDCmodel/Estim/'
directory2 = '/home/jacopo/OneDrive/Dati_Jac_locali/stack/' + qa_name

# server 
directory = 'S:\\users\\jacopo\\Documents\\'
out_dir = 'S:\\Users\\jacopo\\Documents\\out_estim\\'
#directory2 = '\\\\tsclient\\jacopo\\OneDrive\\Dati_Jac_locali\\stack\\apple\\'
directory2 = '\\\\tsclient\\jacopo\\OneDrive\\Dati_Jac_locali\\stack\\ell\\'

# import data
hist = pd.read_csv(directory + 'UserHist_Csmall_week.csv', parse_dates=['day'])

# remove L notation for lag 
hist.columns = [re.sub('L','',i) for i in hist.columns]

# construct for then definition of scarsity
maxavail = np.log(hist['avail']).max() 
    
# design date with change in reputation points
if qa_name == 'ell/':
     designdate = pd.Timestamp(2016,2,25)
     designdateW = pd.date_range(designdate,periods=1, freq='W')[0]
     designdate = hist.loc[hist['day']==designdateW,'datenum'].iloc[0]

# thresholds
thresholds = pd.read_csv(directory2 + 'thresholds.csv')
thresholds = thresholds.loc[(thresholds['type'].notna()) & (thresholds['type']!='Communication')] # some thresholds are not displayed + communiaction thresolds are more complicated: see https://meta.stackexchange.com/questions/58587/what-are-the-reputation-requirements-for-privileges-on-sites-and-how-do-they-di
thresholds = thresholds.loc[thresholds['rep-level']>15] # remove 1 and 10 and 15 thresholds
Tgrad = sorted(thresholds['rep-level'].unique().tolist()[::-1])
Tbeta = sorted(thresholds['rep-level-publicbeta'].unique().tolist()[::-1])

# replace TcumL so to not count threshold = 15 
hist.loc[(hist['datenum']<designdate),'Tcum'] = np.digitize(hist.loc[(hist['datenum']<designdate),'rep_cum'].values, bins=Tbeta)
hist.loc[(hist['datenum']>=designdate),'Tcum'] = np.digitize(hist.loc[(hist['datenum']>=designdate),'rep_cum'].values, bins=Tgrad)

# struct params of first stage estim
delta = 0.95
TTdesigned = 2000
TTbeta = 1000
prob_acceptance = 0.75 # to be estimated with new data
# conversion votes-->points
uppoints = 10 # points per up-vote
downpoints = 2 # points per down-vote
approvalpoints = 2 # points dor approved suggested edits
# rate evolution avail
rateavail = pd.read_pickle(directory2 + 'rate_avail_week.pkl')
# decay params
tau_up = pd.read_pickle(directory2 + 'decay_params_up_week.pkl')[1] # first num is estim of A, second of tau
tau_down = pd.read_pickle(directory2 + 'decay_params_down_week.pkl')[1] # first num is estim of A, second of tau
# reduced-form models
EAE = pd.read_pickle(directory + 'PoissonReg_EdvsQ_noTopics.pkl')
# model: expected upvotes
EUV = pd.read_pickle(directory + 'PoissonReg_UpvsQ_noTopics.pkl')
# model: expected downvotes
EDV = pd.read_pickle(directory + 'PoissonReg_DownvsQ_noTopics.pkl')

### ccp model
states_vars = ['rep_cum','lambda_up','lambda_down','avail','AnswerNum', 'Seniority_days','periods','datenum','Tcum']

if not 'clf_week.pkl' in os.listdir(out_dir):
    scaler = MinMaxScaler()
    scaler = scaler.fit(hist[states_vars].values)
    pd.to_pickle(scaler, out_dir + 'scaler_week.pkl')
    dtahist = scaler.transform(hist[states_vars].values)
    clf = LogisticRegression(solver='saga').fit(X=dtahist, y=hist['choicenum'].values)
    pd.to_pickle(clf, out_dir + 'clf_week.pkl')
    print('ccp model trained')
for t in range(1,4): # for each type
    if not 'clf_week_type{}.pkl'.format(t) in os.listdir(out_dir):
        dtahist = hist.loc[hist['user_types']==t]
        scaler = MinMaxScaler()
        scaler = scaler.fit(dtahist[states_vars].values)
        pd.to_pickle(scaler, out_dir + 'scaler_week_type{}.pkl'.format(t))
        X = scaler.transform(dtahist[states_vars].values)
        Y = dtahist['choicenum'].values
        clf = LogisticRegression(solver='saga').fit(X=X, y=Y)
        pd.to_pickle(clf, out_dir + 'clf_week_type{}.pkl'.format(t))
    

CCPS = {'all':pd.read_pickle(out_dir + 'clf_week.pkl'),
        't1':pd.read_pickle(out_dir + 'clf_week_type1.pkl'),
        't2':pd.read_pickle(out_dir + 'clf_week_type2.pkl'),
        't3':pd.read_pickle(out_dir + 'clf_week_type3.pkl')}

SCALERS = {'all':pd.read_pickle(out_dir + 'scaler_week.pkl'),
           't1':pd.read_pickle(out_dir + 'scaler_week_type1.pkl'),
           't2':pd.read_pickle(out_dir + 'scaler_week_type2.pkl'),
           't3':pd.read_pickle(out_dir + 'scaler_week_type3.pkl')}

# max experience
maxseniority = hist['Seniority_days'].max()
maxanswernum = hist['AnswerNum'].max()

# add isEditor variable
hist.loc[(hist['datenum']<designdate) & (hist['rep_cum']>=TTbeta),'isEditor'] = 1
hist.loc[(hist['datenum']>=designdate) & (hist['rep_cum']>=TTdesigned),'isEditor'] = 1
hist.loc[:,'isEditor'].fillna(0, inplace=True)

# add isDesigned variable
hist.loc[(hist['datenum']<designdate),'isDesigned'] = 0
hist.loc[(hist['datenum']>=designdate),'isDesigned'] = 1

# initialize prob that edit is suggested
hist.loc[:,'editIsSuggested'] = 1 - hist['isEditor'] 

relevant_columns = ['rep_cum','lambda_up','lambda_down','avail','AnswerNum', 'Seniority_days','periods','datenum','Tcum', 'isEditor','editIsSuggested','isDesigned']

# choices
choice_tupl2num = pd.read_pickle(directory2 + 'choice2num_week.pkl')
choices = list(choice_tupl2num.keys())

### ESTIMATION - STEP 1: VARIABLE CONSTRUCTION
# for each possible choice, construct separate versions of the data with the future expected value of states
# but only for states that enter in the utility function
for t in range(1,4):
    datatouse = hist.loc[hist['user_types']==t]
    FES.ExpStates(choices, datatouse, relevant_columns, maxanswernum, maxseniority, maxavail, EAE, EUV,
                  EDV, tau_up, tau_down, rateavail,prob_acceptance, uppoints, downpoints, approvalpoints,
                  Tgrad, Tbeta, TTdesigned, TTbeta, CCPS['t{}'.format(t)], SCALERS['t{}'.format(t)], choice_tupl2num,
                  t, out_dir)
    
### ESTIMATION - STEP 2: MAXIMUM LIKELIHOOD
# ---- Stata
VARS = ['R','CA','CE','Tcum','isEditor','RxE','CAxE','CExE']

for t in range(1,4):
    datatouse = hist.loc[hist['user_types']==t]
    alldfs = []
    for choice in choices:
        choicedf = pd.DataFrame()
        choicestr = '%f_%f_%f'%choice
        datachoice = pd.read_csv(out_dir + 'states_{}_wsparse_week_type{}.csv'.format(choicestr,t), index_col=0)
        datachoice.reset_index(inplace=True)
        
        cv = [i for i in datachoice.columns if re.search('R[0-9]',i)] # just for num of periods
        periods = np.arange(len(cv))
        deltas = delta ** periods
        deltas = np.tile(deltas, (len(datachoice),1))
        
        for var in VARS:
            cv = [i for i in datachoice.columns if re.search('{}[0-9]'.format(var),i)]
    
            var_data = np.sum(datachoice.loc[:,cv].values * deltas, axis=1)
            choicedf.loc[:,var] = var_data
            
        ccpv = [i for i in datachoice.columns if i.startswith('ccp')]
        ccpv_data = np.sum(datachoice.loc[:,ccpv].values * deltas[:,1:], axis=1)
        choicedf.loc[:,'ccp'] = ccpv_data
        
        choicedf.loc[:,'choicenum'] = choice_tupl2num[choice]
        choicedf.loc[:,'user'] = datachoice['user']
        choicedf.loc[:,'periods'] = datachoice['periods']
        choicedf.loc[:,'observations'] = datachoice.index
        
        choicedf = pd.merge(choicedf, datatouse, on=['user','periods'], validate='1:1', how='outer', indicator='merge')
        if any(choicedf['merge']!='both'):
            print('merge mismatch')
            break
    
        alldfs.append(choicedf)
    
    findata = pd.concat(alldfs)
    findata.reset_index(inplace=True, drop=True)
    findata.to_stata(out_dir + 'dfSTATA_week_type{}.dta'.format(t), write_index=False)

# ---- Python

### load all dfs constructed and sort original data
alldfs = {}
for choice in choices:
    choicestr = '%f_%f_%f'%choice
    datachoice = pd.read_csv(out_dir + 'states_%s_wsparse_week.csv'%(choicestr), index_col=0)
    datachoice = datachoice.sort_values(by=['user','periods'])
    alldfs[choice] = datachoice
hist = hist.sort_values(by=['user','periods'])    
    
# first construct value function evaluation for each possible choice
def estim(params, truedata, inputdata):
    diff_cvfs = pd.DataFrame()
    for choice in choices:
        param_vec = params
        choicenum = choice_tupl2num[choice]
        df = inputdata[choice]
    
        if choice == (0,0,0):
            rhoperiods = 0
        elif choice[0]==0 and choice[1]==0:
            rhoperiods = 1
        else:
            inp = pd.DataFrame({'quality':[choice[1]], 'AnswerNum':[maxanswernum],'Seniority_days':[maxseniority]})
            inp['numedits_totalothers_accepted'] = EAE.predict(inp[['quality','AnswerNum','Seniority_days']])  
            meanup = EUV.predict(inp[['quality','AnswerNum','Seniority_days','numedits_totalothers_accepted']])
            rhoperiods = math.ceil((np.log(meanup) -np.log(0.001))*tau_up)

        vars_touse=[]
        ccp_vars = []    
        discount = []
        for period in range(rhoperiods+1):
            vars_touse.extend([i+str(period) for i in VARS])
            discount.append(delta ** period)
            if period != 0: # no ccps in period 0
                ccp_vars.append('ccp'+str(period))
    
        param_vec = np.tile(param_vec, (rhoperiods+1))
        deltavec = np.repeat(discount, len(VARS))
        param_vec = param_vec * deltavec
        
        deltavec_ccps = np.array(discount)[1:]
        summedccps = np.matmul(df[ccp_vars].values,deltavec_ccps)

        diff_cvfs.loc[:,choicenum] = np.matmul(df[vars_touse].values, param_vec) - summedccps
    
    den = np.log(np.sum(np.exp(diff_cvfs), axis=1))
    diff_cvfs['truechoice'] = truedata['choicenum'].values
    diff_cvfs.set_index('truechoice', append=True, inplace=True)
    num = diff_cvfs.stack().reset_index(level=[-1,-2])
    num.rename(columns={'level_2':'choice',0:'num'}, inplace=True)
    num = num.loc[num['truechoice']==num['choice'],'num']
    
    outval = num - den
    outval = - np.sum(outval)
    print(outval)
    return outval

# estim 
VARS = ['R','CA','CE','Tcum','isEditor']
res = sopt.minimize(estim, args=(hist, alldfs), x0=np.repeat(0,len(VARS)), method='BFGS',options={'disp':True, 'maxiter':10000})   
pd.to_pickle(res,out_dir+'res_ell_week.pkl') 
'''
array([ 7.38750202e-03,  4.02209966e-04, -6.13357343e-01, -8.40866304e-01, 1.20524361e+00])
'''
VARS = ['R','CA','CE','Tcum','isEditor','CAxE','CExE']
res2 = sopt.minimize(estim, args=(hist, alldfs),  x0=np.repeat(0,len(VARS)), method='BFGS',options={'disp':True, 'maxiter':10000}) 
pd.to_pickle(res2,out_dir+'res2_ell_week.pkl') 
'''
array([ 6.87488341e-03, -1.00631546e-04, -1.03311125e+01, -7.74517721e-01,
        1.31622958e+00,  6.09261948e-02,  1.22064245e+01])
'''
# output table
res = pd.read_pickle(out_dir+'res_ell_week.pkl')
res2 = pd.read_pickle(out_dir+'res2_ell_week.pkl')
resx = np.append(res.x, [np.nan, np.nan])
a = pd.DataFrame({'Variables':VARS,'Estimates Model 1':np.around(resx,3), 'Estimates Model 2':np.around(res2.x,3)})
a['index'] = np.arange(0,2*len(a),2)
sterrx = np.append(np.sqrt(np.diagonal(res.hess_inv)), [np.nan,np.nan])
b = pd.DataFrame({'Variables':np.repeat(np.nan, len(VARS)),'Estimates Model 1':sterrx, 'Estimates Model 2':np.sqrt(np.diagonal(res2.hess_inv))})
b['index'] = np.arange(1,2*len(b),2)
b.loc[:,'Estimates Model 1'] = b['Estimates Model 1'].apply(lambda x: '({})'.format(round(x,3)))
b.loc[:,'Estimates Model 2'] = b['Estimates Model 2'].apply(lambda x: '({})'.format(round(x,3)))
a = pd.concat([a,b], axis=0)
a = a.sort_values(by='index').drop(columns=['index'])
a.fillna('', inplace=True)
a.replace('(nan)','',inplace=True)
print(a.to_latex(index=False))

'''
  Variables Estimates Model 1 Estimates Model 2
0         R             0.004             0.005
0                       (0.0)             (0.0)
1        CA                 0                 0
1                       (0.0)             (0.0)
2        CE            -4.332             -4.51
2                     (0.002)            (0.26)
3      Tcum            -0.734            -0.718
3                     (0.003)           (0.017)
4  isEditor            -0.019            -0.217
4                     (0.002)           (0.192)
5      CAxE                               0.032
5                                       (0.001)
6      CExE                               0.442
6                                       (0.027)
'''
# only users that passed the threshold
users_passed_threshold = hist.groupby('user')['isEditor'].max()
users_passed_threshold = users_passed_threshold[users_passed_threshold==1]
users_passed_threshold = users_passed_threshold.index.tolist()
reduced_dta = hist.loc[hist['user'].isin(users_passed_threshold)]

alldfs_reduced =  {}
for choice in choices:
    df_ = alldfs[choice]
    alldfs_reduced[choice] = df_.loc[df_['user'].isin(users_passed_threshold)]
    
VARS = ['R','CA','CE','Tcum','isEditor']
res = sopt.minimize(estim, args=(reduced_dta, alldfs_reduced), x0=np.repeat(0,len(VARS)), method='BFGS',options={'disp':True, 'maxiter':10000})   
pd.to_pickle(res,out_dir+'res_ell_week_reduced.pkl') 
'''
array([ 0.00843088,  0.02030536, -2.75357212,  0.7411822 ,  0.1693879 ])
'''

### separately estimate for the different types
# import types
types = pd.read_csv(directory2 + 'individual_chars_dummies_wgroups.csv',usecols=['Id','user_types'])
out_res = {}
for usertype in range(1,4):
    userlist = types.loc[types['user_types']==usertype,'Id'].tolist()
    Tdata = hist.loc[hist['user'].isin(userlist)]
    Idata =  {}
    for choice in choices:
        df_ = alldfs[choice]
        Idata[choice] = df_.loc[df_['user'].isin(userlist)]    
    
    VARS = ['R','CA','CE','Tcum','isEditor']
    res = sopt.minimize(estim, args=(Tdata, Idata), x0=np.repeat(0,len(VARS)), method='BFGS',options={'disp':True, 'maxiter':10000})   
    out_res['res_type{}'.format(usertype)] = res
pd.to_pickle(out_res,out_dir+'res_types.pkl') 
pd.to_pickle(out_res, directory2 + 'DDCmodel\\Estim\\res_types.pkl')

out_res2 = {}
for usertype in range(1,4):
    userlist = types.loc[types['user_types']==usertype,'Id'].tolist()
    Tdata = hist.loc[hist['user'].isin(userlist)]
    Idata =  {}
    for choice in choices:
        df_ = alldfs[choice]
        Idata[choice] = df_.loc[df_['user'].isin(userlist)]    
    
    VARS = ['R','CA','CE','Tcum','isEditor','CAxE','CExE']
    res = sopt.minimize(estim, args=(Tdata, Idata), x0=np.repeat(0,len(VARS)), method='BFGS',options={'disp':True, 'maxiter':10000})   
    out_res2['res_type{}'.format(usertype)] = res
pd.to_pickle(out_res2,out_dir+'res2_types.pkl') 
pd.to_pickle(out_res2, directory2 + 'DDCmodel\\Estim\\res2_types.pkl')

# table res with no interactions - all + types
res = pd.read_pickle(out_dir+'res_ell_week.pkl')
restypes = pd.read_pickle(out_dir+'res_types.pkl')
VARS = ['R','CA','CE','Tcum','isEditor']

a = pd.DataFrame({'Variables':VARS,'Estimates (all sample)':np.around(res.x,4),
                  'Estimates (Type 1)':np.around(restypes['res_type1'].x,4),
                  'Estimates (Type 2)':np.around(restypes['res_type2'].x,4),
                  'Estimates (Type 3)':np.around(restypes['res_type3'].x,4)})
a['index'] = np.arange(0,2*len(a),2)
b = pd.DataFrame({'Estimates (all sample)':np.sqrt(np.diagonal(res.hess_inv)),
                  'Estimates (Type 1)':np.sqrt(np.diagonal(restypes['res_type1'].hess_inv)),
                  'Estimates (Type 2)':np.sqrt(np.diagonal(restypes['res_type2'].hess_inv)),
                  'Estimates (Type 3)':np.sqrt(np.diagonal(restypes['res_type3'].hess_inv))})
b['index'] = np.arange(1,2*len(b),2)
b.loc[:,'(all sample)'] = b['(all sample)'].apply(lambda x: '({})'.format(round(x,4)))
b.loc[:,'(Type 1)'] = b['(Type 1)'].apply(lambda x: '({})'.format(round(x,4)))
b.loc[:,'(Type 2)'] = b['(Type 2)'].apply(lambda x: '({})'.format(round(x,4)))
b.loc[:,'(Type 3)'] = b['(Type 3)'].apply(lambda x: '({})'.format(round(x,4)))

a = pd.concat([a,b], axis=0)
a = a.sort_values(by='index').drop(columns=['index'])
a.fillna('', inplace=True)
print(a.to_latex(index=False))

# table res with interactions - all + types
res = pd.read_pickle(out_dir+'res2_ell_week.pkl')
restypes = pd.read_pickle(out_dir+'res2_types.pkl')
VARS = ['R','CA','CE','Tcum','isEditor','CAxE','CExE']

a = pd.DataFrame({'Variables':VARS,'(all sample)':np.around(res.x,4),
                  '(Type 1)':np.around(restypes['res_type1'].x,4),
                  '(Type 2)':np.around(restypes['res_type2'].x,4),
                  '(Type 3)':np.around(restypes['res_type3'].x,4)})
a['index'] = np.arange(0,2*len(a),2)
b = pd.DataFrame({'(all sample)':np.sqrt(np.diagonal(res.hess_inv)),
                  '(Type 1)':np.sqrt(np.diagonal(restypes['res_type1'].hess_inv)),
                  '(Type 2)':np.sqrt(np.diagonal(restypes['res_type2'].hess_inv)),
                  '(Type 3)':np.sqrt(np.diagonal(restypes['res_type3'].hess_inv))})
b['index'] = np.arange(1,2*len(b),2)
b.loc[:,'(all sample)'] = b['(all sample)'].apply(lambda x: '({})'.format(round(x,4)))
b.loc[:,'(Type 1)'] = b['(Type 1)'].apply(lambda x: '({})'.format(round(x,4)))
b.loc[:,'(Type 2)'] = b['(Type 2)'].apply(lambda x: '({})'.format(round(x,4)))
b.loc[:,'(Type 3)'] = b['(Type 3)'].apply(lambda x: '({})'.format(round(x,4)))

a = pd.concat([a,b], axis=0)
a = a.sort_values(by='index').drop(columns=['index'])
a.fillna('', inplace=True)
print(a.to_latex(index=False))
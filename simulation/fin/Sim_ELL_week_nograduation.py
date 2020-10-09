#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 17:10:41 2019

@author: jacopo
"""

'''
code for the simulation exercise of true data aggregated at weekly level.
U = R + CA + CE + isEditor + RxE + CAxE + CExE
'''
###################################################
#### SIMULATION ###################################
###################################################

import numpy as np
import pandas as pd
import itertools
import scipy.stats as sst
import scipy.sparse as ssp
import os
from numba import jit
import time
from joblib import Parallel, delayed

#from numpy.random import default_rng # not used because in the server there's not intalled
#import shelve
#rng = default_rng()

# qa_name = 'apple/'
qa_name = 'ell/'

directory = '/home/jacopo/OneDrive/Dati_Jac_locali/stack/' + qa_name + 'DDCmodel/Sim_types_nograd/'
basedir = '/home/jacopo/OneDrive/Dati_Jac_locali/stack/' + qa_name
directory2 = '/home/jacopo/OneDrive/Dati_Jac_locali/stack/' + qa_name + 'DDCmodel/Sim_types_nograd/transprob/'

# server
directory = 'S:\\users\\jacopo\\Documents\\Sim_types_nograd\\'
directory2 = 'S:\\users\\jacopo\\Documents\\Sim_types_nograd\\transprob\\'
basedir = '\\\\tsclient\\jacopo\\OneDrive\\Dati_Jac_locali\\stack\\ell\\'


### possible choices
quantity = [0] + list(pd.read_pickle(basedir + 'choice_quantityansw_week.pkl').values)
quality = [0] + list(pd.read_pickle(basedir + 'choice_qualityansw_week.pkl').values)
edits = [0] + list(pd.read_pickle(basedir + 'choice_quantityedits_week.pkl').values)
choices = list(itertools.product(quantity, quality, edits))
toremove = [i for i in choices if i[0]==0 and i[1]!=0] # if quantity is 0, then quality cannot be positive
toremove2 = [ i for i in choices if i[1]==0 and i[0]!=0] # and viceversa
for choice in toremove+toremove2:
    choices.remove(choice)

### state vars
VARS = ['R','CA','CE','Tcum','isEditor','CAxE','CExE']

### parameters
typenum = 3
if typenum == 'all':
    res = pd.read_pickle(basedir + 'DDCmodel/Estim/res2_ell_week.pkl')
    params = res.x
else:
    res = pd.read_pickle(basedir + 'DDCmodel/Estim/res2_types.pkl')
    params = res['res_type{}'.format(typenum)].x

rand_state_dict = {1:111,
                   2:211,
                   3:311,
                   'all':411}
rand_state = rand_state_dict[typenum]
num_periods = 100
num_users = 100

### create panel dataframe

userids = [i for i in range(num_users)]
periods = [i for i in range(num_periods)]
panels = list(itertools.product(userids,periods))

simdata = pd.DataFrame(panels, columns=['user','periods'])

### initial values for endogenous and exogenous states
simdata.loc[simdata['periods']==0,'rep_cum'] = 0
simdata.loc[simdata['periods']==0,'lambda_up'] = 0
simdata.loc[simdata['periods']==0,'lambda_down'] = 0
simdata.loc[simdata['periods']==0,'AnswerNum'] = 0
#simdata.loc[simdata['periods']==0,'AnswerNumEffective'] = 0 # this is different from AnswerNum if the state space of AnswerNum has been limited for comput. reasons
simdata.loc[simdata['periods']==0,'Seniority_days'] = 0

# datenum variable
dates2num = pd.read_pickle(basedir + 'date2num_week.pkl')
# -- > random initial date
dates2num = dates2num.iloc[:-num_periods] # to be sure that time series doesn't exit period of data
daysdata = dates2num.sample(num_users, replace=True, random_state=rand_state) 
simdata.loc[simdata['periods']==0,'day'] = daysdata['day'].values
simdata.loc[simdata['periods']==0,'datenum'] = daysdata['datenum'].values
# -- > all start the same day, not long before change in priv level
#arb_day = pd.Timestamp(2016,2,1) # arbitrary day (here taken to be shortly before change in threshold values)
#arb_day_W = pd.date_range(arb_day,periods=1, freq='W')[0] 
#simdata.loc[simdata['periods']==0,'day'] = arb_day_W
def daterange(start):
    return pd.date_range(start, start+pd.Timedelta(num_periods-1, unit='W'), freq='W')
simdata.loc[:,'day'] = simdata.groupby('user')['day'].transform(lambda x: daterange(x.iloc[0]))


### add axogenous variable 'availability'
availability = pd.read_csv(basedir + 'availability.csv',index_col=0, parse_dates=True )
# complete index
compl_avail_index = pd.date_range(start=availability.index.min(), end=availability.index.max(), freq='D')
availability = availability.reindex(compl_avail_index, method='ffill')
availability.rename(columns=lambda x: 'avail_'+x, inplace=True)
availabilityW = availability.resample('W').mean()
simdata = pd.merge(simdata, availability, left_on='day', right_index=True, how='left', validate='m:1')
# allocate expertise
expertise = pd.read_csv(basedir + 'expertise.csv')
random_exp = expertise.sample(num_users,replace=False, random_state=rand_state)
u = np.arange(num_users)
np.random.seed(rand_state); np.random.shuffle(u)
random_exp.loc[:,'OwnerUserId'] = u
random_exp.rename(columns={'OwnerUserId':'user'},inplace=True)
simdata = pd.merge(simdata, random_exp, on='user',validate='m:1', how='left')
# create avail
topics = [i for i in random_exp.columns if i!='user']
avtopics = ['avail_'+i for i in topics]

avail = np.multiply(simdata[avtopics],simdata[topics]) 
avail = np.sum(avail, axis=1)

simdata.loc[:,'avail'] = avail

simdata.drop(availability.columns.tolist() + topics, axis=1, inplace=True)


### reduce dimensions of state space for computationl burden - only for value functions and utilities, not for transition
# avail - to riduce dimensionality to 5 bins (with value = to min of bin)
vals = np.linspace(0,availability.iloc[:,1:5].max().max(),5)
simdata.loc[:,'avail'] = pd.cut(simdata['avail'], bins=vals, right=True, include_lowest=True)
simdata.loc[:,'avail'] = simdata['avail'].apply(lambda x: np.round(x.mid,0))
simdata.loc[:,'avail'] = simdata['avail'].astype(int)

# max avail for later use 
maxavail = np.log(simdata['avail']).max() 


### struct params of first stage estim
delta = 0.95
TTdesigned = 500
prob_acceptance = 0.75
# conversion votes-->points
uppoints = 5 # points per up-vote
downpoints = 1 # points per down-vote
approvalpoints = 1 # points for approved suggested edits
# rate evolution avail
rateavail = pd.read_pickle(basedir + 'rate_avail_week.pkl')
# decay params
tau_up = pd.read_pickle(basedir + 'decay_params_up_week.pkl')[1] # first num is estim of A, second of tau
tau_down = pd.read_pickle(basedir + 'decay_params_down_week.pkl')[1] # first num is estim of A, second of tau
# reduced-form models
EAE = pd.read_pickle(basedir + 'PoissonReg_EdvsQ_noTopics.pkl')
# model: expected upvotes
EUV = pd.read_pickle(basedir + 'PoissonReg_UpvsQ_noTopics.pkl')
# model: expected downvotes
EDV = pd.read_pickle(basedir + 'PoissonReg_DownvsQ_noTopics.pkl')
# # evolution of accepted answers
# EccA_params = pd.read_pickle(basedir + 'evolv_aceptedanswers_params.pkl')

# thresholds
thresholds = pd.read_csv(basedir + 'thresholds.csv')
thresholds = thresholds.loc[(thresholds['type'].notna()) & (thresholds['type']!='Communication')] # some thresholds are not displayed + communiaction thresolds are more complicated: see https://meta.stackexchange.com/questions/58587/what-are-the-reputation-requirements-for-privileges-on-sites-and-how-do-they-di
thresholds = thresholds.loc[thresholds['rep-level']>15] # remove 1 and 10 and 15 thresholds
Tgrad = sorted(thresholds['rep-level'].unique().tolist()[::-1])
# divide by 2 since amount of points per up/down vote is dividedby 2
Tgrad = [round(i/2) for i in Tgrad]

### possible values of states
# rep
rep_max = 1500
possible_points = np.arange(0,rep_max,1)
# lambda up
lambda_up_max = 0.2
possible_lambdaup = np.round(np.arange(0,lambda_up_max,0.01),2)
# lambda down
lambda_down_max = 0.03
possible_lambdadown = np.round(np.arange(0,lambda_down_max,0.01),2)
# avail
possible_avail = np.sort(simdata['avail'].unique())
# answer num
an_max = 1
possible_answernum = np.arange(0,an_max,1)
# seniority
sen_max = 1
possible_seniority = np.arange(0,sen_max,1)



# create all possible combinations of states

states = list(itertools.product(possible_points, possible_lambdaup,possible_lambdadown,possible_avail,
                                possible_answernum,possible_seniority))

state_names = ['rep_cum','lambda_up','lambda_down','avail','AnswerNum', 'Seniority_days']

states = pd.DataFrame(np.array(states), columns = state_names)
statesreset = states.reset_index()


### utlity function 
def computeFlow_lowmem(st, ch, params): 
    if type(ch)!=np.ndarray:
        ch = np.array(ch)
        
    R = np.tile(st['rep_cum'].values[:,np.newaxis], (1,len(ch))) 
    CA = (ch[:,[0]]**(maxavail / np.log(st['avail'].values)) + ch[:,[1]]).T# dim n*j
    CE = ch[:,[2]]
    CE = np.tile(CE, (1,len(st))).T  # dim n*j

    Tcum = np.digitize(st['rep_cum'].values, Tgrad)
    Tcum = np.tile(Tcum[:,np.newaxis], (1,len(ch)))
    E = np.tile(st['isEditor'].values[:,np.newaxis], (1,len(ch))) # dim 1* (n*j)
    CAxE = CA * E
    CExE = CE * E 

    u = (R * params[0] + CA * params[1] + CE * params[2] + Tcum * params[3] + E * params[4] 
         + CAxE * params[5] + CExE * params[6])         
    return u     


### transition matrix
''' note: building sparse matrices, if multiple values are assigned to the same cell,
  then the values are summed. See the example below, where the bottom right cell is assigned twice:
row = np.array([0, 0, 1, 2, 2, 2, 2])
col = np.array([0, 2, 2, 0, 1, 2, 2])
data = np.array([1, 2, 3, 4, 5, 6, 1])
csr_matrix((data, (row, col)), shape=(3, 3)).toarray()
'''     

stdata = states.copy()


stdata.loc[:,'isEditor'] = np.where(stdata['rep_cum']>=TTdesigned, 1, 0)
stdata['editIsSuggested'] = 1 - stdata['isEditor']

print('computing transition probabilities')

#for choice in choices:
def TPmatrix(choice):
    # if '%f_%f_%f.npz'%(choice[0],choice[1],choice[2]) in os.listdir(directory2):
    #     continue
    start = time.time()
    #print('choice',choice,'started in period', period)
    rows_ind = [] 
    cols_ind = [] 
    data_val = [] 

    stdata.loc[:,'quality'] = choice[1]
    stdata.loc[:,'numedits_totalothers_accepted'] = EAE.predict(stdata[['quality','AnswerNum','Seniority_days']]) 
    meanup_main = stdata['lambda_up'] + choice[0]*EUV.predict(stdata[['quality','AnswerNum','Seniority_days','numedits_totalothers_accepted']])
    meandown_main = stdata['lambda_down'] +  choice[0]*EDV.predict(stdata[['quality','AnswerNum','Seniority_days','numedits_totalothers_accepted']])
    fvup_main = np.arange(sst.poisson.ppf(0.001, mu=meanup_main.min()),sst.poisson.ppf(0.999, mu=meanup_main.max())+1) # plus one to include right boundary (as it was done in simultion)
    fvdown_main = np.arange(sst.poisson.ppf(0.001, mu=meandown_main.min()),sst.poisson.ppf(0.999, mu=meandown_main.max())+1) # plus one to include right boundary (as it was done in simultion)
    accedits_main = np.arange(0,choice[2]+1) # number of suggested edits potentially approved

    extra_main_new_up = ssp.csc_matrix(uppoints*fvup_main)
    colsup = np.repeat([i for i in range(len(fvup_main))], len(fvdown_main)*len(accedits_main))
    extra_main_new_up = extra_main_new_up[:,colsup]
    
    extra_main_new_down = ssp.csc_matrix(downpoints*fvdown_main)
    colsdown = np.tile(np.repeat([i for i in range(len(fvdown_main))], len(accedits_main)), (len(fvup_main)))
    extra_main_new_down = extra_main_new_down[:,colsdown]
    
    extra_main_new_edits = ssp.csc_matrix(approvalpoints*accedits_main)
    colsedits = np.tile([i for i in range(len(accedits_main))], (int(extra_main_new_up.shape[1]/len(accedits_main))))
    extra_main_new_edits = extra_main_new_edits[:,colsedits]
    
    extra_main_new = extra_main_new_up - extra_main_new_down + extra_main_new_edits
    rows = np.repeat(0, len(stdata), axis=0)
    extra_main_new = extra_main_new[rows,:]
    
    # probabilities of new points arriving / of values at a)
    tpup_main_new =  ssp.csc_matrix(sst.poisson.pmf(fvup_main,mu=meanup_main[:,np.newaxis]))
    tpup_main_new = tpup_main_new[:,colsup] # use column indeces created before
    
    tpdown_main_new =  ssp.csc_matrix(sst.poisson.pmf(fvdown_main,mu=meandown_main[:,np.newaxis]))
    tpdown_main_new = tpdown_main_new[:,colsdown]
    
    tpedits_main = ssp.csc_matrix(sst.binom.pmf(accedits_main, choice[2], prob_acceptance*stdata['editIsSuggested'][:,np.newaxis]))
    tpedits_main = tpedits_main[:,colsedits]
    
    tp_main_new = tpup_main_new.multiply(tpdown_main_new).multiply(tpedits_main)    
    
    # reduce dimensionality
    fut_vals_main = extra_main_new[0,:].toarray()[0,:]
    unique_fut_vals_main = np.unique(fut_vals_main)
    tp_reduced_main = ssp.lil_matrix((extra_main_new.shape[0],unique_fut_vals_main.shape[0]))
    cols_tokeep = [] # columns to keep in extra_main
    count = 0 # to keep order of columns in new tp_main
    for val in list(unique_fut_vals_main):
        colsval = np.where(fut_vals_main==val)[0]
        cols_tokeep.append(colsval[0]) # collects one column for each value to retain from extra_main
        tp_reduced_main[:,count] = tp_main_new[:,colsval].sum(axis=1)
        count += 1
    tp_main_new = tp_reduced_main.tocsr()
    extra_main_new = extra_main_new[:,cols_tokeep]

    fut_dta = pd.DataFrame()
    fut_dta['lambda_up'] = np.minimum(np.around(meanup_main * np.exp(-1/tau_up),2), max(possible_lambdaup))
    fut_dta['lambda_down'] = np.minimum(np.around(meandown_main * np.exp(-1/tau_down),2), max(possible_lambdadown))
    fut_dta['AnswerNum'] = np.minimum(stdata['AnswerNum'] + choice[0], max(possible_answernum))
    fut_dta['avail'] = stdata['avail'] # no beliefs on evolution for now
    fut_dta['Seniority_days'] = np.minimum(stdata['Seniority_days'] + 1, max(possible_seniority))

    for col in range(extra_main_new.shape[1]):
        fut_dta.loc[:,'rep_cum'] = np.maximum(np.minimum(stdata['rep_cum'][:,np.newaxis] + extra_main_new[:,col].toarray(), max(possible_points)), 0)[:,0]
        prob = tp_main_new[:,col].toarray()
        fut_dta = pd.merge(fut_dta, statesreset, on=state_names, how='left')
        rows_ind.extend(fut_dta.index.tolist())
        cols_ind.extend(fut_dta['index'].values.tolist()) 
        data_val.extend(prob.flatten().tolist())             
        fut_dta.drop('index', axis=1, inplace=True)
    print(time.time()-start) 
    matrix = ssp.csr_matrix((data_val,(rows_ind,cols_ind)),shape=(len(stdata),len(stdata)))
  
    ssp.save_npz(directory2 + '%f_%f_%f.npz'%(choice[0],choice[1],choice[2]), matrix)
    print('choice {} completed'.format(choice))
Parallel(n_jobs=6)(delayed(TPmatrix)(choice) for choice in choices)
    

### value functions
        
print('computing value functions')

vf = {}
# last period
stdata = states.copy()
#stdata = pd.DataFrame(stdata, columns = state_names)
#stdata['isEditor'] = np.where(stdata['rep_cum']>=TT,1,0)

stdata.loc[:,'isEditor'] = np.where(stdata['rep_cum']>=TTdesigned, 1, 0)

U = computeFlow_lowmem(stdata, choices, params)
err = np.random.gumbel(size= len(choices))
err = np.tile(err, (len(stdata),1))
U = U + err
vmax = np.max(U, axis=1)
vf[num_periods-1] = vmax


for period in range(max(periods)-1,0,-1):
    print('started period',period)

    #stdata.loc[:,'isEditor'] = np.where(stdata['rep_cum']>=TTdesigned, 1, 0)
    
    U = computeFlow_lowmem(stdata, choices, params)
    err = np.random.gumbel(size= len(choices))
    err = np.tile(err, (len(stdata),1))
    U = U + err
    
    EVF = ssp.lil_matrix((len(stdata),len(choices)))
    for choice in choices:
        tp = ssp.load_npz(directory2 + '%f_%f_%f'%choice + '.npz')
        EVF[:,choices.index(choice)] = delta * tp.dot(vf[period+1])[:,np.newaxis]
    U = U + EVF.toarray()
    vmax = np.max(U, axis=1)
    vf[period] = vmax
pd.to_pickle(vf, directory + 'vf_type{}.pkl'.format(typenum))

#vf = pd.read_pickle(directory + 'vf_type{}.pkl'.format(typenum))

### forward simulate

print('simulation started')

for period in range(num_periods):
    print('started period',period)


    simdata.loc[simdata['periods']==period,'isEditor'] = np.where(simdata.loc[simdata['periods']==period,'rep_cum'].values>=TTdesigned, 1, 0)

    states = simdata.loc[simdata['periods']==period,]
    
    U = computeFlow_lowmem(states, choices, params)
    err = np.random.gumbel(size= (len(states),len(choices)))# shock is different for each user and each choice
    U = U + err

    states = pd.merge(states, statesreset, on=state_names, how='left')
    
    if period+1 != num_periods:
        EVF = ssp.lil_matrix((len(states),len(choices)))

        for choice in choices:
            tp = ssp.load_npz(directory2 + '%f_%f_%f'%choice + '.npz')
            tp = tp[states['index'],:]
            EVF[:,choices.index(choice)] = delta * tp.dot(vf[period+1])[:,np.newaxis]
        U = U + EVF.toarray()        
    
    ch_matrix = np.tile(np.array([i for i in range(len(choices))]), [len(states),1])
    c_star = np.where(U==np.max(U, axis=1)[:,np.newaxis], ch_matrix, 0)
    if any([i>1 for i in np.count_nonzero(c_star,axis=1)]):
        print('Error, indifference between choices')
        break
    c_star = c_star.sum(axis=1)

    states['choice'] = c_star # add in temp data
    states['quantity'] = states['choice'].apply(lambda x: choices[int(x)][0])
    states['quality'] = states['choice'].apply(lambda x: choices[int(x)][1])
    states['numedits'] = states['choice'].apply(lambda x: choices[int(x)][2])

    # add optimal choices    
    simdata.loc[simdata['periods']==period,'choice'] = c_star # add in final data
    simdata.loc[simdata['periods']==period,'numAnswers'] = states['quantity'].values
    simdata.loc[simdata['periods']==period,'quality'] = states['quality'].values
    simdata.loc[simdata['periods']==period,'numEdits'] = states['numedits'].values
    
    # transition of states: only for period different from the last
    if period+1 == num_periods:
        break
    
    states.loc[:,'numedits_totalothers_accepted'] = EAE.predict(states[['quality','AnswerNum','Seniority_days']])
    meanup = states['lambda_up'] + states['quantity']*EUV.predict(states[['quality','AnswerNum','Seniority_days','numedits_totalothers_accepted']])
    meandown = states['lambda_down'] + states['quantity']*EDV.predict(states[['quality','AnswerNum','Seniority_days','numedits_totalothers_accepted']])
    
    upvotes = np.random.poisson(meanup)
    downvotes = np.random.poisson(meandown)
    accedits = np.random.binomial(states['numedits'], prob_acceptance*(1-states['isEditor']))
    
    points = uppoints * upvotes - downpoints * downvotes + approvalpoints * accedits
    
    # fill in simulated dataframe
    simdata.loc[simdata['periods']==period+1,'rep_cum'] = np.maximum(np.minimum(states['rep_cum'].values + points, max(possible_points)), 0)
    simdata.loc[simdata['periods']==period+1,'lambda_up'] = np.minimum(np.around(meanup.values * np.exp(-1/tau_up),2), max(possible_lambdaup))
    simdata.loc[simdata['periods']==period+1,'lambda_down'] = np.minimum(np.around(meandown.values * np.exp(-1/tau_down),2), max(possible_lambdadown))
    simdata.loc[simdata['periods']==period+1,'Seniority_days'] = np.minimum(states['Seniority_days'].values + 1, max(possible_seniority))
    simdata.loc[simdata['periods']==period+1,'AnswerNum'] = np.minimum(states['AnswerNum'].values + states['quantity'].values, max(possible_answernum))
    

simdata.to_csv(directory + 'simdata_type{}.csv'.format(typenum), index=False)
simdata.to_csv(basedir + 'DDCmodel\\Sim_types_nograd\\simdata_type{}.csv'.format(typenum), index=False)
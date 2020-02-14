#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 11:39:40 2020

@author: jacopo
"""

'''
new approach to estimation with finite dependence
'''

import pandas as pd
import numpy as np
import scipy.sparse as ssp
import scipy.stats as sst
from sklearn.linear_model import LogisticRegression
import itertools

directory = '/home/jacopo/OneDrive/Dati_Jac_locali/stack/apple/'
directory2 = '/home/jacopo/OneDrive/Dati_Jac_locali/stack/apple/DDCmodel/'
directory_mat = directory2 + 'TransitionMatrices_SimNobins/'
dir_futurestates = '/home/jacopo/OneDrive/Dati_Jac_locali/stack/apple/DDCmodel/dfs_futurestates/'

directory = '\\\\tsclient\\jacopo\\OneDrive\\Dati_Jac_locali\\stack\\apple\\'
directory2 = '\\\\tsclient\\jacopo\\OneDrive\\Dati_Jac_locali\\stack\\apple\\DDCmodel\\'
directory_mat = directory2 + 'TransitionMatrices_SimNobins\\'
dir_futurestates = directory2 + 'dfs_futurestates\\'

# import data
truedata = pd.read_csv(directory2 + 'simdata_nobins.csv', index_col=0)

# ### load relevant objects
# # all possible states
# allstates_df = pd.read_csv(directory2 + 'allstates_df_noUchars.csv')
# allstates = allstates_df.columns.tolist()
# tom = allstates_df.reset_index()
# truedata = pd.merge(truedata, tom, on=allstates, how='left')
# # binning mapping
# num2intXstate = pd.read_pickle(directory2 + 'num2intXstate_noUchars.pkl')
# num2intXchoice = pd.read_pickle(directory2 + 'num2intXchoice_noUchars.pkl')

### possible choices
achoice = [0,1,2]
echoice = [0,1,2]
choices = list(itertools.product(achoice, echoice))
### ccp model
states_vars = ['rep_cum', 'expUVLAG']
clf = LogisticRegression(solver='liblinear').fit(X=truedata[states_vars+['period']].values, y=truedata['choice'].values)

# struct params of first stage estim
zeta = 0.5#0.13296
# p_acceptance = 0.75
# delta = 0.95
tau = 25 # privilege threshold

VARS = ['points','Achoice','Echoice']

# def valueInInt(x, st):
#     for int_num, interval in num2intXstate[st].items():
#         if x in interval:
#             return int_num
# binEditorGrad = valueInInt(2000, 'rep_cum')

# set max values per states ( in real data it's inf so do not set)
maxrep = 50
maxexpuvlag = 105

# for each possible choice, construct separate versions of the data with the future expected value of states
# but only for states that enter in the utility function
for choice in choices:
    
    print('started choice',choice)
    choicestr = str(choice[0]) +'_'+ str(choice[1])
    
    # df to store final variables
    dff_out = truedata[['userid','period']]
    
    # df to store state values for each choice
    dff_main = truedata.copy()
    dff_main.loc[:,'choice'] = choicestr
    dff_main.rename(columns=lambda x:x+'0', inplace=True)

    dff_ref = truedata.copy()
    dff_ref.loc[:,'choice'] = choicestr
    dff_ref.rename(columns=lambda x:x+'0', inplace=True)

    # set number rho of periods ahead are necessary, given choice
    rhoperiods = int(choice[0] / 0.5)
    # choice sequences
    c_main = [choice] + [(0,0) for i in range(rhoperiods)]
    c_ref =[(0,0),choice] + [(0,0) for i in range(1,rhoperiods)]
    
    # tp0 = ssp.load_npz(directory_mat + '0_0.npz')
    # tpchoice = ssp.load_npz(directory_mat + choicestr + '.npz')
    
    # tps_main = {}
    # tps_ref = {}
    # # store transition probabilities
    # for period in range(1,rhoperiods+1):
    #     if period == 1:
    #         tps_main[period] = tpchoice
    #         tps_ref[period] = tp0
    #     elif period == 2:
    #         tps_main[period] = tps_main[1].dot(tp0)
    #         tps_ref[period] = tps_ref[1].dot(tpchoice)
    #     else:
    #         tps_main[period] = tps_main[period-1].dot(tp0)
    #         tps_ref[period] = tps_ref[period-1].dot(tp0)            

    
    # period 0          
    dff_out.loc[:,'R0'] = 0 # points are the same between reference and main path in period 0
    dff_out.loc[:,'A0'] = c_main[0][0] - c_ref[0][0]
    dff_out.loc[:,'E0'] = c_main[0][1] - c_ref[0][1]
    dff_out.loc[:,'Re0']= 0
    dff_out.loc[:,'Ae0'] = (c_main[0][0] - c_ref[0][0]) * dff_main['Editor0'] # here dff_main or dff_ref are indifferent
    dff_out.loc[:,'Ee0'] = (c_main[0][1] - c_ref[0][1]) * dff_main['Editor0']
    
    extra_main = np.array([[0]])
    tp_main = np.array([[1]])
    extra_ref = np.array([[0]])
    tp_ref = np.array([[1]])
    for period in range(1,rhoperiods+1):
        
        ### construct state variables
        # assume 1 upvote --> 1 point
        # MAIN
        mean_main = np.maximum(dff_main['expUVLAG%d'%(period-1)] - zeta,0) + c_main[period-1][0] 
        fv_main = np.arange(sst.poisson.ppf(0.001, mu=mean_main.min()),sst.poisson.ppf(0.999, mu=mean_main.max()))

        # future extra points
        extra_main_new =  np.tile(fv_main, (len(dff_main),1)) # possible new points for each state
        extra_main_new = np.repeat(extra_main_new, extra_main.shape[1], axis=1) # possible new points given past points
        extra_main = np.tile(extra_main, (1,len(fv_main))) # reshape old extra points 
        extra_main = extra_main + extra_main_new
        totextra_main = extra_main + dff_main['rep_cum0'][:,np.newaxis] # possible final rep points
        totextra_main = np.minimum(totextra_main, maxrep) # set max number of points user can get
        
        # probabilities
        tp_main_new = sst.poisson.pmf(fv_main,mu=mean_main[:,np.newaxis])
        tp_main_new = np.repeat(tp_main_new, tp_main.shape[1], axis=1)
        tp_main = np.tile(tp_main, (1,len(fv_main))) # reshape old extra points 
        tp_main = tp_main * tp_main_new
        
        dff_main.loc[:,'rep_cum%d'%(period)] = np.sum(tp_main * totextra_main, axis=1)
        
        tpe_main = np.where(totextra_main>=tau, tp_main, 0)
        dff_main.loc[:,'rep_cumE%d'%(period)] = np.sum(tpe_main * totextra_main, axis=1)
        
        dff_main['expUVLAG%d'%(period)] = np.minimum(mean_main, maxexpuvlag)
        
        # REF
        mean_ref = np.maximum(dff_ref['expUVLAG%d'%(period-1)] - zeta,0) + c_ref[period-1][0] 
        fv_ref = np.arange(sst.poisson.ppf(0.001, mu=mean_ref.min()),sst.poisson.ppf(0.999, mu=mean_ref.max()))

        # future extra points
        extra_ref_new =  np.tile(fv_ref, (len(dff_ref),1)) # possible new points for each state
        extra_ref_new = np.repeat(extra_ref_new, extra_ref.shape[1], axis=1) # possible new points given past points
        extra_ref = np.tile(extra_ref, (1,len(fv_ref))) # reshape old extra points 
        extra_ref = extra_ref + extra_ref_new
        totextra_ref = extra_ref + dff_ref['rep_cum0'][:,np.newaxis] # possible final rep points
        totextra_ref = np.minimum(totextra_ref, maxrep) # set max number of points user can get
        
        # probabilities
        tp_ref_new = sst.poisson.pmf(fv_ref,mu=mean_ref[:,np.newaxis])
        tp_ref_new = np.repeat(tp_ref_new, tp_ref.shape[1], axis=1)
        tp_ref = np.tile(tp_ref, (1,len(fv_ref))) # reshape old extra points 
        tp_ref = tp_ref * tp_ref_new
        
        dff_ref.loc[:,'rep_cum%d'%(period)] = np.sum(tp_ref * totextra_ref, axis=1)
        
        tpe_ref = np.where(totextra_ref>=tau, tp_ref, 0)
        dff_ref.loc[:,'rep_cumE%d'%(period)] = np.sum(tpe_ref * totextra_ref, axis=1)
        
        dff_ref['expUVLAG%d'%(period)] = np.minimum(mean_ref, maxexpuvlag)
        
        dff_out.loc[:,'R%d'%(period)] = dff_main['rep_cum%d'%(period)] - dff_ref['rep_cum%d'%(period)]
        dff_out.loc[:,'Re%d'%(period)] = dff_main['rep_cumE%d'%(period)] - dff_ref['rep_cumE%d'%(period)]

        dff_out.loc[:,'A%d'%(period)] = c_main[period][0] - c_ref[period][0]
        dff_out.loc[:,'E%d'%(period)] = c_main[period][1] - c_ref[period][1]
        dff_out.loc[:,'Ae%d'%(period)] = c_main[period][0] * np.sum(tpe_main, axis=1) - c_ref[period][0] * np.sum(tpe_ref, axis=1)
        dff_out.loc[:,'Ee%d'%(period)] = c_main[period][1] * np.sum(tpe_main, axis=1) - c_ref[period][1] * np.sum(tpe_ref, axis=1) 
        
        ## ccps
        
        # MAIN
        dff_main.loc[:,'period%d'%(period)] = dff_main['period0'] + period
        # predicted probabilities
        ccps_main = [] # compute ccps fo each possible future states, then take expectation
        for i in range(totextra_main.shape[1]):
            futdata = np.array([totextra_main[:,i], dff_main['expUVLAG%d'%(period)].values, dff_main['period%d'%(period)]]).T
            p = clf.predict_log_proba(futdata)[:,choices.index(choice)]
            ccps_main.append(p)
        ccps_main = np.array(ccps_main).T 
        dff_main.loc[:,'ccp%d'%(period)] = np.sum(tp_main * ccps_main, axis=1) # expected value of ccp
            
        # REF
        dff_ref.loc[:,'period%d'%(period)] = dff_ref['period0'] + period
        # predicted probabilities
        ccps_ref = [] # compute ccps fo each possible future states, then take expectation
        for i in range(totextra_ref.shape[1]):
            futdata = np.array([totextra_ref[:,i], dff_ref['expUVLAG%d'%(period)].values, dff_ref['period%d'%(period)]]).T
            p = clf.predict_log_proba(futdata)[:,choices.index(choice)]
            ccps_ref.append(p)
        ccps_ref = np.array(ccps_ref).T 
        dff_ref.loc[:,'ccp%d'%(period)] = np.sum(tp_ref * ccps_ref, axis=1) # expected value of ccp
             
            
        dff_out.loc[:,'ccp%d'%(period)] = dff_main['ccp%d'%(period)] - dff_ref['ccp%d'%(period)]
    dff_out.to_csv(dir_futurestates + 'states_%sT.csv'%(choicestr))

def load_dfs():
    alldfs = {}
    for choice in choices:
        choicestr = str(choice[0]) +'_'+ str(choice[1])
        alldfs[choice] = pd.read_csv(dir_futurestates + 'states_%s.csv'%(choicestr), index_col=0)
    return alldfs

def Ufunction(params):
    states_dfs = load_dfs()
    # first construct value function evaluation for each possible choice
    vfs = pd.DataFrame()
    for choice in choices:
        
    
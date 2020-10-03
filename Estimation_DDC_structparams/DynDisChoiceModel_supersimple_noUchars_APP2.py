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
    
    extra_main = ssp.csr_matrix( np.repeat([[0]],len(dff_main), axis=0))
    tp_main = ssp.csr_matrix( np.repeat([[1]],len(dff_main), axis=0))
    extra_ref = ssp.csr_matrix( np.repeat([[0]],len(dff_ref), axis=0))
    tp_ref = ssp.csr_matrix( np.repeat([[1]],len(dff_ref), axis=0))
    for period in range(1,rhoperiods+1):
        
        ### construct state variables
        # assume 1 upvote --> 1 point
        # MAIN
        mean_main = np.maximum(dff_main['expUVLAG%d'%(period-1)] - zeta,0) + c_main[period-1][0] 
        fv_main = np.arange(sst.poisson.ppf(0.001, mu=mean_main.min()),sst.poisson.ppf(0.999, mu=mean_main.max()))
        fv_main = fv_main[np.newaxis,:]
        # future extra points: create 2 overlapping matrix, one for all possible future points, one with the prob for each of them
        '''
        say that, given period 0 action, in period 1 you could get 1 or 2 extra points, and
        given period 1 action, 2 or 3 extra points in period 2.
        Then I build:
            - from matrix [2,3] --> [2,2,3,3] witn nrows = len of true data
            - from previous period matrix [1,2]--> [1,2,1,2] witn nrows = len of true data
            - i sum the two, to et all possible combinations of final extra points: [3,4,4,5] witn nrows = len of true data
            - same to construct probabilities:
            - from matrix of period 2 [p2(2),p2(3)] and period 1 matrix [p1(1),p1(2)] i construct
            - [p2(2)p1(1) , p2(2)p1(2), p2(3)p1(1), p2(3)p1(2)]
            - then i multiply the matrix of extra values element by elment with the probs and sum 
            - horizontally to get expected value.
        '''
        # a) new points arriving
        extra_main_new = ssp.csr_matrix(fv_main)
        rows = np.repeat(0, len(dff_main), axis=0)
        cols = np.repeat([i for i in range(fv_main.shape[1])], extra_main.shape[1])
        extra_main_new = extra_main_new[rows,:] # slice in separate commands to be sure it works fine
        extra_main_new = extra_main_new[:,cols]
        
        # b) final num points arriving summing all periods after period 0
        cols = np.tile([i for i in range(extra_main.shape[1])], (1,fv_main.shape[1]))[0]
        extra_main = extra_main[:,cols]
        extra_main = extra_main + extra_main_new
        
        # c) final cumulated reputation obtained after all periods after period 0
        repl = np.repeat(0 ,extra_main.shape[1]) # in sparse matrices it's only possible to sum matrices of same dimension
        totextra_main = extra_main + ssp.csr_matrix(dff_main['rep_cum0'][:,np.newaxis])[:,repl] # possible final rep points
        totextra_main.data = np.minimum(totextra_main.data, maxrep) # set max number of points user can get
        
        # probabilities of new points arriving / of values at a)
        tp_main_new = sst.poisson.pmf(fv_main,mu=mean_main[:,np.newaxis])
        tp_main_new = ssp.csr_matrix(tp_main_new)
        
        # joint probabilities given past / ot values at b) and c)
        cols = np.repeat([i for i in range(fv_main.shape[1])], tp_main.shape[1])
        tp_main_new = tp_main_new[:,cols]
        cols = np.tile([i for i in range(tp_main.shape[1])], (1,fv_main.shape[1]))[0]
        tp_main = tp_main[:,cols]
        tp_main = tp_main.multiply(tp_main_new)
        
        dff_main.loc[:,'rep_cum%d'%(period)] = tp_main.multiply(totextra_main).sum(axis=1)

        # construct matrix of 1 and zeros, 1 if user is Editor
        totextraEbin_main = totextra_main.copy()
        totextraEbin_main.data = np.where(totextra_main.data>=tau, 1, 0)
        
        # values at c, but setting to zero if user is not yet an editor
        totextraE_main = totextra_main.multiply(totextraEbin_main)
        dff_main.loc[:,'rep_cumE%d'%(period)] = tp_main.multiply(totextraE_main).sum(axis=1)
        
        dff_main['expUVLAG%d'%(period)] = np.minimum(mean_main, maxexpuvlag)
        
        # Variables A*Editor and E*Editor
        dff_main['AxE%d'%(period)] = totextraEbin_main.multiply(c_main[period][0]).multiply(tp_main).sum(axis=1)
        dff_main['ExE%d'%(period)] = totextraEbin_main.multiply(c_main[period][1]).multiply(tp_main).sum(axis=1)
        
        # REF
        mean_ref = np.maximum(dff_ref['expUVLAG%d'%(period-1)] - zeta,0) + c_ref[period-1][0] 
        fv_ref = np.arange(sst.poisson.ppf(0.001, mu=mean_ref.min()),sst.poisson.ppf(0.999, mu=mean_ref.max()))
        fv_ref = fv_ref[np.newaxis,:]
        # future extra points: create 2 overlapping matrix, one for all possible future points, one with the prob for each of them

        # a) new points arriving
        extra_ref_new = ssp.csr_matrix(fv_ref)
        rows = np.repeat(0, len(dff_ref), axis=0)
        cols = np.repeat([i for i in range(fv_ref.shape[1])], extra_ref.shape[1])
        extra_ref_new = extra_ref_new[rows,:] # slice in separate commands to be sure it works fine
        extra_ref_new = extra_ref_new[:,cols]
        
        # b) final num points arriving summing all periods after period 0
        cols = np.tile([i for i in range(extra_ref.shape[1])], (1,fv_ref.shape[1]))[0]
        extra_ref = extra_ref[:,cols]
        extra_ref = extra_ref + extra_ref_new
        
        # c) final cumulated reputation obtained after all periods after period 0
        repl = np.repeat(0 ,extra_ref.shape[1]) # in sparse matrices it's only possible to sum matrices of same dimension
        totextra_ref = extra_ref + ssp.csr_matrix(dff_ref['rep_cum0'][:,np.newaxis])[:,repl] # possible final rep points
        totextra_ref.data = np.minimum(totextra_ref.data, maxrep) # set max number of points user can get
        
        # probabilities of new points arriving / of values at a)
        tp_ref_new = sst.poisson.pmf(fv_ref,mu=mean_ref[:,np.newaxis])
        tp_ref_new = ssp.csr_matrix(tp_ref_new)
        
        # joint probabilities given past / ot values at b) and c)
        cols = np.repeat([i for i in range(fv_ref.shape[1])], tp_ref.shape[1])
        tp_ref_new = tp_ref_new[:,cols]
        cols = np.tile([i for i in range(tp_ref.shape[1])], (1,fv_ref.shape[1]))[0]
        tp_ref = tp_ref[:,cols]
        tp_ref = tp_ref.multiply(tp_ref_new)
        
        dff_ref.loc[:,'rep_cum%d'%(period)] = tp_ref.multiply(totextra_ref).sum(axis=1)

        # construct matrix of 1 and zeros, 1 if user is Editor
        totextraEbin_ref = totextra_ref.copy()
        totextraEbin_ref.data = np.where(totextra_ref.data>=tau, 1, 0)
        
        # values at c, but setting to zero if user is not yet an editor
        totextraE_ref = totextra_ref.multiply(totextraEbin_ref)
        dff_ref.loc[:,'rep_cumE%d'%(period)] = tp_ref.multiply(totextraE_ref).sum(axis=1)
        
        dff_ref['expUVLAG%d'%(period)] = np.minimum(mean_ref, maxexpuvlag)

        # Variables A*Editor and E*Editor
        dff_ref['AxE%d'%(period)] = totextraEbin_ref.multiply(c_ref[period][0]).multiply(tp_ref).sum(axis=1)
        dff_ref['ExE%d'%(period)] = totextraEbin_ref.multiply(c_ref[period][1]).multiply(tp_ref).sum(axis=1)

        
        dff_out.loc[:,'R%d'%(period)] = dff_main['rep_cum%d'%(period)] - dff_ref['rep_cum%d'%(period)]
        dff_out.loc[:,'Re%d'%(period)] = dff_main['rep_cumE%d'%(period)] - dff_ref['rep_cumE%d'%(period)]

        dff_out.loc[:,'A%d'%(period)] = c_main[period][0] - c_ref[period][0]
        dff_out.loc[:,'E%d'%(period)] = c_main[period][1] - c_ref[period][1]
        
        dff_out.loc[:,'AxE%d'%(period)] = dff_main['AxE%d'%(period)] - dff_ref['AxE%d'%(period)]
        dff_out.loc[:,'ExE%d'%(period)] = dff_main['ExE%d'%(period)] - dff_ref['ExE%d'%(period)]
        
        ## ccps
        
        # MAIN
        dff_main.loc[:,'period%d'%(period)] = dff_main['period0'] + period
        # predicted probabilities
        ccps_main = ssp.lil_matrix((len(dff_main),totextra_main.shape[1])) # compute ccps fo each possible future states and add to sparse matrix, then take expectation
        for i in range(totextra_main.shape[1]):
            futdata = np.array([totextra_main[:,i].toarray().flatten(), dff_main['expUVLAG%d'%(period)], dff_main['period%d'%(period)]]).T
            p = clf.predict_log_proba(futdata)[:,choices.index(choice)]
            ccps_main[:,i] = p[:,np.newaxis]
        ccps_main = ccps_main.tocsr()
        dff_main.loc[:,'ccp%d'%(period)] = tp_main.multiply(ccps_main).sum(axis=1) # expected value of ccp

        # REF
        dff_ref.loc[:,'period%d'%(period)] = dff_ref['period0'] + period
        # predicted probabilities
        ccps_ref = ssp.lil_matrix((len(dff_ref),totextra_ref.shape[1])) # compute ccps fo each possible future states and add to sparse matrix, then take expectation
        for i in range(totextra_ref.shape[1]):
            futdata = np.array([totextra_ref[:,i].toarray().flatten(), dff_ref['expUVLAG%d'%(period)], dff_ref['period%d'%(period)]]).T
            p = clf.predict_log_proba(futdata)[:,choices.index(choice)]
            ccps_ref[:,i] = p[:,np.newaxis]
        ccps_ref = ccps_ref.tocsr()
        dff_ref.loc[:,'ccp%d'%(period)] = tp_ref.multiply(ccps_ref).sum(axis=1) # expected value of ccp
             
    
        dff_out.loc[:,'ccp%d'%(period)] = dff_main['ccp%d'%(period)] - dff_ref['ccp%d'%(period)]
        
    dff_out.to_csv(dir_futurestates + 'states_%s_wsparse.csv'%(choicestr))

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
        
    
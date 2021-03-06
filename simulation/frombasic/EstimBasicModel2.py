#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 11:39:40 2020

@author: jacopo
"""

'''
Estimation for simulated data created with SimBasicModel2.py
'''

import pandas as pd
import numpy as np
import scipy.sparse as ssp
import scipy.stats as sst
from sklearn.linear_model import LogisticRegression
import scipy.optimize as sopt

directory = '/home/jacopo/OneDrive/Dati_Jac_locali/stack/apple/DDCmodel/SimBaseModel2_files/'

# import data
simdata = pd.read_csv(directory + 'simdataBasicModel2.csv', index_col=0)

### possible choices
choices = [0,1,2]

### ccp model
states_vars = ['rep_cum', 'expUVLAG','avail']
clf = LogisticRegression(solver='liblinear').fit(X=simdata[states_vars+['period']].values, y=simdata['choice'].values)

# struct params of first stage estim
zeta = 0.5
delta = 0.95

# set max values per states ( in real data it's inf so do not set)
maxrep = 100
maxexpuvlag = 105

# for exogenous state ('avail') construct future values + drop last periods for which are not know enough periods ahead
maxchoice = max(choices)
maxrhoperiods = int(maxchoice / 0.5)
for period in range(1,maxrhoperiods+1):
    simdata['avail%d'%(period)] = simdata.groupby('userid')['avail'].transform(lambda x: x.shift(periods=-period))    
simdata.rename(columns={'avail':'avail0'}, inplace=True)
maxperiods = simdata['period'].max()
simdata = simdata.loc[simdata['period']<=maxperiods-maxrhoperiods]

# for each possible choice, construct separate versions of the data with the future expected value of states
# but only for states that enter in the utility function
for choice in choices:
    
    print('started choice',choice)
    choicestr = str(choice)
    
    # df to store final variables
    dff_out = simdata[['userid','period']]
    
    # df to store state values for each choice
    dff_main = simdata.copy()
    dff_main.loc[:,'choice'] = choicestr
    dff_main.rename(columns=lambda x:x+'0' if not x.startswith('avail') else x, inplace=True)

    dff_ref = simdata.copy()
    dff_ref.loc[:,'choice'] = choicestr
    dff_ref.rename(columns=lambda x:x+'0' if not x.startswith('avail') else x, inplace=True)

    # set number rho of periods ahead are necessary, given choice
    rhoperiods = int(choice / 0.5)
    # choice sequences
    c_main = [choice] + [0 for i in range(rhoperiods)]
    c_ref =[0,choice] + [0 for i in range(1,rhoperiods)]          
    
    # period 0          
    dff_out.loc[:,'R0'] = 0 # points are the same between reference and main path in period 0
    dff_out.loc[:,'A0'] = (c_main[0] - c_ref[0]) / dff_main['avail0'] # avail is the same btw main and ref

    # initialize matrices for later use
    extra_main = ssp.csr_matrix( np.repeat([[0]],len(dff_main), axis=0))
    tp_main = ssp.csr_matrix( np.repeat([[1]],len(dff_main), axis=0))
    extra_ref = ssp.csr_matrix( np.repeat([[0]],len(dff_ref), axis=0))
    tp_ref = ssp.csr_matrix( np.repeat([[1]],len(dff_ref), axis=0))
    for period in range(1,rhoperiods+1):
        
        ### construct state variables
        # MAIN
        mean_main = np.maximum(dff_main['expUVLAG%d'%(period-1)] - zeta,0) + c_main[period-1]
        fv_main = np.arange(sst.poisson.ppf(0.001, mu=mean_main.min()),sst.poisson.ppf(0.999, mu=mean_main.max())+1) # plus one to include right boundary (as it was done in simultion)
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
                
        # probabilities of new points arriving / of values at a)
        tp_main_new = sst.poisson.pmf(fv_main,mu=mean_main[:,np.newaxis])
        tp_main_new = ssp.csr_matrix(tp_main_new)
        
        # joint probabilities given past / of values at b)
        cols = np.repeat([i for i in range(fv_main.shape[1])], tp_main.shape[1])
        tp_main_new = tp_main_new[:,cols]
        cols = np.tile([i for i in range(tp_main.shape[1])], (1,fv_main.shape[1]))[0]
        tp_main = tp_main[:,cols]
        tp_main = tp_main.multiply(tp_main_new)

        # reduce dimensionality
        fut_vals_main = extra_main[0,:].toarray()[0,:]
        unique_fut_vals_main = np.unique(fut_vals_main)
        tp_reduced_main = ssp.lil_matrix((extra_main.shape[0],unique_fut_vals_main.shape[0]))
        cols_tokeep = [] # columns to keep in extra_main
        count = 0 # to keep order of columns in new tp_main
        for val in list(unique_fut_vals_main):
            cols = np.where(fut_vals_main==val)[0]
            cols_tokeep.append(cols[0]) # collects one column for each value to retain from extra_main
            tp_reduced_main[:,count] = tp_main[:,cols].sum(axis=1)
            count += 1
        tp_main = tp_reduced_main.tocsr()
        extra_main = extra_main[:,cols_tokeep]

        # c) final cumulated reputation obtained after all periods after period 0
        repl = np.repeat(0 ,extra_main.shape[1]) # in sparse matrices it's only possible to sum matrices of same dimension
        totextra_main = extra_main + ssp.csr_matrix(dff_main['rep_cum0'][:,np.newaxis])[:,repl] # possible final rep points
        totextra_main.data = np.minimum(totextra_main.data, maxrep) # set max number of points user can get
        
        dff_main.loc[:,'rep_cum%d'%(period)] = tp_main.multiply(totextra_main).sum(axis=1)
                
        dff_main['expUVLAG%d'%(period)] = np.minimum(mean_main, maxexpuvlag)
                
        # REF
        mean_ref = np.maximum(dff_ref['expUVLAG%d'%(period-1)] - zeta,0) + c_ref[period-1]
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
                
        # probabilities of new points arriving / of values at a)
        tp_ref_new = sst.poisson.pmf(fv_ref,mu=mean_ref[:,np.newaxis])
        tp_ref_new = ssp.csr_matrix(tp_ref_new)
        
        # joint probabilities given past / ot values at b) and c)
        cols = np.repeat([i for i in range(fv_ref.shape[1])], tp_ref.shape[1])
        tp_ref_new = tp_ref_new[:,cols]
        cols = np.tile([i for i in range(tp_ref.shape[1])], (1,fv_ref.shape[1]))[0]
        tp_ref = tp_ref[:,cols]
        tp_ref = tp_ref.multiply(tp_ref_new)

        # reduce dimensionality
        fut_vals_ref = extra_ref[0,:].toarray()[0,:]
        unique_fut_vals_ref = np.unique(fut_vals_ref)
        tp_reduced_ref = ssp.lil_matrix((extra_ref.shape[0],unique_fut_vals_ref.shape[0]))
        cols_tokeep = [] # columns to keep in extra_ref
        count = 0 # to keep order of columns in new tp_ref
        for val in list(unique_fut_vals_ref):
            cols = np.where(fut_vals_ref==val)[0]
            cols_tokeep.append(cols[0]) # collects one column for each value to retain from extra_ref
            tp_reduced_ref[:,count] = tp_ref[:,cols].sum(axis=1)
            count += 1
        tp_ref = tp_reduced_ref.tocsr()
        extra_ref = extra_ref[:,cols_tokeep]

        # c) final cumulated reputation obtained after all periods after period 0
        repl = np.repeat(0 ,extra_ref.shape[1]) # in sparse matrices it's only possible to sum matrices of same dimension
        totextra_ref = extra_ref + ssp.csr_matrix(dff_ref['rep_cum0'][:,np.newaxis])[:,repl] # possible final rep points
        totextra_ref.data = np.minimum(totextra_ref.data, maxrep) # set max number of points user can get
        
        dff_ref.loc[:,'rep_cum%d'%(period)] = tp_ref.multiply(totextra_ref).sum(axis=1)
        
        dff_ref['expUVLAG%d'%(period)] = np.minimum(mean_ref, maxexpuvlag)
        
        dff_out.loc[:,'R%d'%(period)] = dff_main['rep_cum%d'%(period)] - dff_ref['rep_cum%d'%(period)]
        dff_out.loc[:,'A%d'%(period)] = (c_main[period] - c_ref[period]) / dff_main['avail%d'%(period)]  # avail is the same btw main and ref
        

        ## ccps
        
        # MAIN
        dff_main.loc[:,'period%d'%(period)] = dff_main['period0'] + period
        # predicted probabilities
        ccps_main = ssp.lil_matrix((len(dff_main),totextra_main.shape[1])) # compute ccps fo each possible future states and add to sparse matrix, then take expectation
        for i in range(totextra_main.shape[1]):
            futdata = np.array([totextra_main[:,i].toarray().flatten(), dff_main['expUVLAG%d'%(period)],dff_main['avail%d'%(period)], dff_main['period%d'%(period)]]).T
            p = clf.predict_log_proba(futdata)
            p = pd.DataFrame(p, columns=clf.classes_)
            p = p.loc[:,choices.index(c_main[period])].values
            ccps_main[:,i] = p[:,np.newaxis]
        ccps_main = ccps_main.tocsr()
        dff_main.loc[:,'ccp%d'%(period)] = tp_main.multiply(ccps_main).sum(axis=1) # expected value of ccp

        # REF
        dff_ref.loc[:,'period%d'%(period)] = dff_ref['period0'] + period
        # predicted probabilities
        ccps_ref = ssp.lil_matrix((len(dff_ref),totextra_ref.shape[1])) # compute ccps fo each possible future states and add to sparse matrix, then take expectation
        for i in range(totextra_ref.shape[1]):
            futdata = np.array([totextra_ref[:,i].toarray().flatten(), dff_ref['expUVLAG%d'%(period)],dff_main['avail%d'%(period)], dff_ref['period%d'%(period)]]).T
            p = clf.predict_log_proba(futdata)
            p = pd.DataFrame(p, columns=clf.classes_)
            p = p.loc[:,choices.index(c_ref[period])].values
            ccps_ref[:,i] = p[:,np.newaxis]
        ccps_ref = ccps_ref.tocsr()
        dff_ref.loc[:,'ccp%d'%(period)] = tp_ref.multiply(ccps_ref).sum(axis=1) # expected value of ccp
             
    
        dff_out.loc[:,'ccp%d'%(period)] = dff_main['ccp%d'%(period)] - dff_ref['ccp%d'%(period)]
        
    dff_out.to_csv(directory + 'states_%s_wsparse.csv'%(choicestr))

### load all dfs constructed
alldfs = {}
for choice in choices:
    choicestr = str(choice)
    alldfs[choice] = pd.read_csv(directory + 'states_%s_wsparse.csv'%(choicestr), index_col=0)


# param_vec_init = np.array([beta_rep,beta_effA,beta_effE,beta_repXEditor,beta_effAXEditor,beta_effEXEditor])

VARS = ['R','A']

# mapping choice num to choice str
choice_str2num = {}
count =0
for choice in choices:
    choicestr = str(choice)
    choice_str2num[choicestr] = count
    count += 1
    
# first construct value function evaluation for each possible choice
def estim(params):
    diff_cvfs = pd.DataFrame()
    for choice in choices:
        param_vec = params
        choicestr = str(choice) 
        df = alldfs[choice]
    
        rhoperiods = int(choice/ 0.5)
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

        diff_cvfs.loc[:,choicestr] = np.matmul(df[vars_touse].values, param_vec) - summedccps
    
    x = np.abs(diff_cvfs).max().max()
    if x > 700:
        print('adgustment of support')
        if diff_cvfs.max().max() > 700:
            diff_cvfs = diff_cvfs - (x - 700)
        else:
            diff_cvfs = diff_cvfs + (x - 700)
    
    den = np.log(np.sum(np.exp(diff_cvfs), axis=1))
    diff_cvfs['truechoice'] = simdata['choice'].values
    diff_cvfs.set_index('truechoice', append=True, inplace=True)
    num = diff_cvfs.stack().reset_index(level=[-1,-2])
    num.rename(columns={'level_2':'choice',0:'num'}, inplace=True)
    num.loc[:,'choice'] = num['choice'].apply(lambda x: choice_str2num[x])
    num.loc[:,'truechoice'] = num['truechoice'].astype('int')
    num = num.loc[num['truechoice']==num['choice'],'num']
    
    outval = num - den
    outval = - np.sum(outval)
    print(outval)
    return outval
        

sopt.minimize(estim, x0=np.repeat(0,2), method='BFGS',options={'disp':True, 'maxiter':10000})    
'''
      fun: 1122.8469256559165
 hess_inv: array([[0.00199466, 0.00155894],
       [0.00155894, 0.0016469 ]])
      jac: array([0.00000000e+00, 1.52587891e-05])
  message: 'Desired error not necessarily achieved due to precision loss.'
     nfev: 37
      nit: 7
     njev: 9
   status: 2
  success: False
        x: array([ 0.09518411, -0.26136356])
'''

sopt.minimize(estim, x0=np.repeat(0,2), method='Nelder-Mead',options={'disp':True, 'maxiter':10000})    
'''
 final_simplex: (array([[ 0.09520531, -0.26136478],
       [ 0.09517643, -0.2612923 ],
       [ 0.09518532, -0.26144987]]), array([1122.84692617, 1122.84692676, 1122.84692723]))
           fun: 1122.8469261748796
       message: 'Optimization terminated successfully.'
          nfev: 96
           nit: 50
        status: 0
       success: True
             x: array([ 0.09520531, -0.26136478])
'''
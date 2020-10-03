#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 17:10:41 2019

@author: jacopo
"""

'''
code for the simulation exercise.
This is the simulation of a basic model where there is no threshold, only points and cost of answering.

The model here is anyway different from SimBasicModel.py, because the cost of effort depends on the topic available
(i.e. if there is a lot of questions the person knows the answer or only few)
call this variable "availability"

The model here is different from SimBasicModel2.py because the evolution of the expUVLAG is now multiplicative.

The model here is different form SimBasicModel3.py because the evolution of points depends on a non-choice specific state.
Identification of the parameter of the cost was allowed by having the availability variable which is varying cost but not points.
Here is then to see if having a variable impacting uniquely the evolution of points helps even more to separately
identify the coefficients of cost ad points.


U = beta0*R + beta1*Cost + epsilon

where Cost == A / availability

(let r_t be the new points arrived in period t)
where E_t[R_t+1] = R_t + E_t[r_t+1] = R_t + lambda_t+1 = R_t + zeta*lambda_t + K
K is the 1-period return from effort. In the earlier SimBaseModeln.py (n<=3) K was = A.
Now K == A + w*1{A>0} (w is added only if positive effort)
where w is how many new questions pop up in the topic of experize of the agent. w can be 0, 1, 2.

availability can be 0.5, 1, or 1.5, so that very low levels makes bigger costs and 

HERE THE VARIABLE AVAILABILITY IS PERFECTLY KNOWN AND FORECASTED BY THE AGENT. THE AGENT OBSERVES FROM THE
BEGINNING THE EVOLUTION OF THE EXOGENOUS VARIABLE. 

The variable w evolves according to the following transition:
w_t+1 = 
    w_t + 1 if w_t < 2
    0  if w_t == 2

'''

import numpy as np
import pandas as pd
import itertools
import scipy.stats as sst
import scipy.sparse as ssp

directory = '/home/jacopo/OneDrive/Dati_Jac_locali/stack/apple/DDCmodel/SimBaseModel4_files/'

### possible choices
choices = [0,1,2] # only effort in answering

# struct params of first stage estim
zeta = 0.5 # just for simulation and check that estimation works
delta = 0.95
points_perupvote = 1 # assume 1 in simulation instead of 10

### panel dimensions
num_periods = 30
num_users = 50

###################################################
#### SIMULATION ###################################
###################################################

### create panel dataframe

userids = [i for i in range(num_users)]
periods = [i for i in range(num_periods)]
panels = list(itertools.product(userids,periods))

simdata = pd.DataFrame(panels, columns=['userid','period'])

### parameters

# for now consider variables: points, effort_A, effort_E
beta_rep = 0.1
beta_cost = -0.3

params = np.array([beta_rep,beta_cost])

### initial values for endogenous and exogenous states
simdata.loc[simdata['period']==0,'rep_cum'] = 0
simdata.loc[simdata['period']==0,'expUVLAG'] = 0
simdata.loc[:,'avail'] = np.random.choice([0.5,1,1.5], size=len(simdata))
simdata.loc[simdata['period']==0,'w'] = np.random.choice([0,1,2], size=num_users)

def transition_w(x):
    x = x.values
    newdta = np.repeat(0,num_periods)
    for obsnum in range(len(x)):
        if obsnum == 0:
            newdta[obsnum] = x[obsnum]
        elif newdta[obsnum-1] == 2:
            newdta[obsnum] = 0
        else:
            newdta[obsnum] = newdta[obsnum-1] + 1
    return newdta
simdata.loc[:,'w'] = simdata.groupby('userid')['w'].transform(lambda x: transition_w(x))


### obtain value functions for each possible state, in each possible period
### starting from the last

def computeFlow(st, ch, params):
    
    cost =  np.repeat(ch,len(st)) / st['avail']
    
    dta = np.array([st['rep_cum'],cost]).T
    
    u = np.matmul(dta, params)
    
    return u  


# compute value functions for all possible values of the endogenous state, at each period

# assume you can get up to 100 points
possible_points = [i for i in range(101)]
# recover possible range for expUVLAG (max is case in which achoice is max in all periods)
max_possible_expuvlag = 0
for p in range(1,num_periods+1):
    max_possible_expuvlag = round(2* zeta * max_possible_expuvlag)/2 + max(choices) + simdata['w'].max() # this way of rounding is distorting, don't use in real data
    print(max_possible_expuvlag)
 # assume simplified version to have discrete expUVLAG
possible_expuvlag = [i for i in np.arange(0,max_possible_expuvlag+0.5, 0.5)]
possible_w = [0,1,2]
st_data = list(itertools.product(possible_points,possible_expuvlag,possible_w))
st_data = pd.DataFrame(st_data, columns=['rep_cum','expUVLAG','w'])    

# create transition matrices
states_vars = ['rep_cum', 'expUVLAG','w']
index_mapping = st_data[states_vars].reset_index()
index_mapping.set_index(states_vars, inplace=True)
index_mapping = index_mapping['index']
index_mapping = index_mapping.to_dict()
''' note: building sparse matrices, if multiple values are assigned to the same cell,
 then the values are summed. See the example below, where the bottom right cell is assigned twice:
row = np.array([0, 0, 1, 2, 2, 2, 2])
col = np.array([0, 2, 2, 0, 1, 2, 2])
data = np.array([1, 2, 3, 4, 5, 6, 1])
csr_matrix((data, (row, col)), shape=(3, 3)).toarray()
'''     
for choice in choices:
    print('choice',choice,'started')
    rows = [] 
    cols = [] 
    data = [] 
    for row_num, state in st_data.iterrows():
        mean = round(2* zeta * state['expUVLAG'])/2 + choice + state['w']*np.sign(choice)
        fv = np.arange(sst.poisson.ppf(0.001, mu=mean),sst.poisson.ppf(0.999, mu=mean)+1)
        fp = sst.poisson.pmf(fv,mu=mean)
        tp = pd.DataFrame({'fut_points':fv,'prob':fp})
        tp['fut_repcum'] = np.minimum(tp['fut_points'] + state['rep_cum'], max(st_data['rep_cum']))
        tp['fut_expuvlag'] = np.minimum(mean, max(st_data['expUVLAG']))
        if state['w']==2:
            tp['fut_w'] = 0
        else:
            tp['fut_w'] = int(state['w'] + 1)

        tp['row_indexes'] = row_num
        tp['col_indexes'] = tp.apply(lambda row: index_mapping[(row['fut_repcum'],row['fut_expuvlag'],row['fut_w'])], axis=1)

        rows.extend(tp['row_indexes'].values.tolist())
        cols.extend(tp['col_indexes'].values.tolist()) 
        data.extend(tp['prob'].values.tolist()) 
    matrix = ssp.csr_matrix((data,(rows,cols)),shape=(len(st_data),len(st_data)))
    
    ssp.save_npz(directory + '%d.npz'%(choice), matrix)


vfs = {}
# last period
vfT = []
# replicate all possible states for each user
allust_data = pd.DataFrame(np.tile(st_data.values,(num_users,1)), columns=st_data.columns)
allust_data.loc[:,'avail'] = np.repeat(simdata.loc[simdata['period']==num_periods-1, 'avail'].values, len(st_data))
for iteration in range(2): # average max utility across shocks
    v = []
    for c in choices:
        v.append(computeFlow(allust_data, c, params) + np.random.gumbel())
    v = np.array(v)
    # same shock for all states, but choice specific
    vmax = np.max(v, axis=0)
    vfT.append(vmax)
vfT = np.array(vfT)
vmax = np.mean(vfT, axis=0)
vfs[max(periods)] = vmax


for period in range(max(periods)-1,0,-1):
    vfT = []
    allust_data = pd.DataFrame(np.tile(st_data.values,(num_users,1)), columns=st_data.columns)
    allust_data.loc[:,'avail'] = np.repeat(simdata.loc[simdata['period']==period, 'avail'].values, len(st_data))
    for iteration in range(2):
        v = []
        for c in choices:
            err = np.random.gumbel()
            tp = ssp.load_npz(directory + '%s.npz'%(str(c)))
            tp = ssp.block_diag([tp for i in range(num_users)],format='csr')
            v.append(computeFlow(allust_data, c, params) + delta * tp.dot(vfs[period+1]) + err)
        # same shock for all states, but choice specific
        v = np.array(v)
        vmax = np.max(v, axis=0)
        vfT.append(vmax)
    vfT = np.array(vfT)
    vmax = np.mean(vfT, axis=0)
    vfs[period] = vmax

pd.to_pickle(vfs, directory + 'vfs_simBase4.pkl')
#vfs = pd.read_pickle(directory2 + 'vfs_simBase3.pkl')

### forward simulate

# to recover state indexes

for period in range(num_periods):
    states = simdata.loc[simdata['period']==period,]
    
    cvfs = []
    for c in choices:

        # try to select from transition prob matrix only inital states appearing in states
        tp = ssp.load_npz(directory + '%s.npz'%(str(c)))
        states['state_index'] = states.apply(lambda row: index_mapping[(row['rep_cum'],row['expUVLAG'],row['w'])], axis=1)
        tp = tp[states['state_index']]
        tp = ssp.block_diag([tp[i,] for i in range(tp.shape[0])],format='csr')

        if period+1 == num_periods:
            cvfs.append(computeFlow(states, c, params)
                        + np.fromiter((np.random.gumbel() for i in range(len(states))),float))
                        # shock is different for each user
        else:
            cvfs.append(computeFlow(states, c, params) + 
                        delta * tp.dot(vfs[period+1]) 
                        + np.fromiter((np.random.gumbel() for i in range(len(states))),float))
                        # shock is different for each user
    v = np.array(cvfs).T
    ch_matrix = np.tile(np.array([i for i in range(len(choices))]), [len(states),1])
    c_star = np.where(v==np.max(v, axis=1)[:,np.newaxis], ch_matrix, 0)
    if any([i>1 for i in np.count_nonzero(c_star,axis=1)]):
        print('Error, indifference between choices')
        break
    c_star = c_star.sum(axis=1)
    # add optimal choices    
    simdata.loc[simdata['period']==period,'choice'] = c_star # add in final data
    states['choice'] = c_star # add in temp data
    
    # transition of states: only for period different from the last
    if period+1 == num_periods:
        break

    states.loc[:,'expUVLAG_t+1'] = round(zeta*2*states['expUVLAG'])/2 + states['choice'] + states['w']
    states.loc[:,'rep_cum_t+1'] = np.minimum(states['expUVLAG_t+1'].apply(lambda x: np.random.poisson(x)) + states['rep_cum'],max(possible_points))

        
    # fill in simulated dataframe
    simdata.loc[simdata['period']==period+1,'rep_cum'] = states['rep_cum_t+1'].values
    simdata.loc[simdata['period']==period+1,'expUVLAG'] = states['expUVLAG_t+1'].values
    
simdata.to_csv(directory + 'simdataBasicModel4.csv')



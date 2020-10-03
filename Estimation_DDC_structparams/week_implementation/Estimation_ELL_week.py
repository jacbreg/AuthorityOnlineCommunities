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
reciprocity for answers = cost_answeing * cum tot num of answers received on owned questions
reciprocity on editing = cost_editing * cum num of implemented edits received on own answers (recall: includes rallbacks)
altruism answering = cost_answeing * cum num of asnwers given selected as "accepted answer"
altruism editing = cost_editing * cum num of upvotes receveid by other people's answers after own edits'

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
    #- evolution of arrived answers to own questions--> model based on num questions published and num answers already received
    #- evolution of arrived edits to own answers --> model used to predic mean
    #- altruism variable--> expectation of not increase over time
        
'''
import pandas as pd
import numpy as np
import scipy.sparse as ssp
import scipy.stats as sst
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import scipy.optimize as sopt
import itertools
import math
from collections import Counter
import re
import os
#from numpy.random import default_rng
#import sys
#rng = default_rng()

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

# day num
if not 'date2num_week.pkl' in os.listdir(directory2):
    dates = pd.date_range(start=hist['day'].min(), end=hist['day'].max(), freq='1W')
    dates2num = pd.DataFrame({'day':dates,'datenum':np.arange(len(dates))})
    pd.to_pickle(dates2num, directory2 + 'date2num_week.pkl')
else:
    dates2num = pd.read_pickle(directory2 + 'date2num_week.pkl')
    
hist = pd.merge(hist, dates2num, on='day', validate='m:1')

# construct for then definition of scarsity
maxavail = np.log(hist['avail']).max() 
    
hist.loc[:,'choice'] = hist.apply(lambda row: (row['numAnswers'],row['quality'],row['numEdits']), axis=1)

choices = hist['choice'].unique()
choices = np.sort(choices)

# choice to num dict
if not 'choice2num_week.pkl' in os.listdir(out_dir) or not 'num2choice_week.pkl' in os.listdir(out_dir): # do this only with full dataset
    choice_tupl2num = {}
    count =0
    for choice in choices:
        choice_tupl2num[choice] = count
        count +=1
    
    choice_num2tupl = {value:key for key, value in choice_tupl2num.items()}
    pd.to_pickle(choice_num2tupl, out_dir + 'num2choice_week.pkl')
    pd.to_pickle(choice_tupl2num, out_dir + 'choice2num_week.pkl')
else:
    choice_num2tupl = pd.read_pickle(out_dir + 'num2choice_week.pkl')
    choice_tupl2num = pd.read_pickle(out_dir + 'choice2num_week.pkl')

hist.loc[:,'choicenum'] = hist['choice'].apply(lambda x: choice_tupl2num[x])

# design date with change in reputation points
if qa_name == 'ell/':
     designdate = pd.Timestamp(2016,2,25)
     designdateW = pd.date_range(designdate,periods=1, freq='W')[0]
     designdate = dates2num.loc[dates2num['day']==designdateW,'datenum'].iloc[0]

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
# # model: expected received answers (only parameters)
# ERA_params = pd.read_pickle(directory2 + 'beliefs_receivedanswers_params.pkl')
# def ERA_predict(received_answers_cum, numQuestions_cum, params):
#     return np.exp(ERA_params['Intercept'] + ERA_params['received_answers_cum']* received_answers_cum + ERA_params['numQuestions_cum']*numQuestions_cum)

### ccp model
scaler = MinMaxScaler()
#hist.loc[:,'logavail'] = np.log(hist['avail'])
states_vars = ['rep_cum','lambda_up','lambda_down','avail','AnswerNum', 'Seniority_days','periods','datenum','Tcum']
#hist = hist.loc[(hist['lambda_up'].notna()) & (hist['lambda_down'].notna())]
scaler = scaler.fit(hist[states_vars].values)

if not 'clf_week.pkl' in os.listdir(out_dir):
    dtahist = scaler.transform(hist[states_vars].values)
    clf = LogisticRegression(solver='saga').fit(X=dtahist, y=hist['choicenum'].values)
    pd.to_pickle(clf, out_dir + 'clf_week.pkl')
    print('ccp model trained')
else:
    clf = pd.read_pickle(out_dir + 'clf_week.pkl')

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

# for each possible choice, construct separate versions of the data with the future expected value of states
# but only for states that enter in the utility function
for choice in choices:
    
    print('started choice',choice)
    choicestr = '%f_%f_%f'%choice
    
    # df to store final variables
    dff_out = hist[['user','periods']]
    
    # df to store state values for each choice
    dff_main = hist[relevant_columns].copy()
    #dff_main.loc[:,'choice'] = choicestr
    #dff_main.rename(columns=lambda x:x+'0', inplace=True)

    dff_ref = hist[relevant_columns].copy()
    #dff_ref.loc[:,'choice'] = choicestr
    #dff_ref.rename(columns=lambda x:x+'0', inplace=True)

    # set number rho of periods ahead are necessary, given choice
    if choice == (0,0,0):
        rhoperiods = 0
    elif choice[0]==0 and choice[1]==0:
        rhoperiods = 1
    else:
        inp = pd.DataFrame({'quality':[choice[1]], 'AnswerNum':[maxanswernum],'Seniority_days':[maxseniority]})
        inp['numedits_totalothers_accepted'] = EAE.predict(inp[['quality','AnswerNum','Seniority_days']])  
        meanup = EUV.predict(inp[['quality','AnswerNum','Seniority_days','numedits_totalothers_accepted']])
        rhoperiods = math.ceil((np.log(meanup) -np.log(0.001))*tau_up)
    # choice sequences
    c_main = [choice] + [(0,0,0) for i in range(rhoperiods)]
    c_ref =[(0,0,0),choice] + [(0,0,0) for i in range(1,rhoperiods)]          
    
    # period 0          
    dff_out.loc[:,'R0'] = 0 # points are the same between reference and main path in period 0
    dff_out.loc[:,'CA0'] = ((c_main[0][0]**(maxavail / np.log(dff_main['avail'])) + c_main[0][1])-(c_ref[0][0]**(maxavail / np.log(dff_main['avail'])) + c_ref[0][1])) # avail is the same btw main and ref
    dff_out.loc[:,'CE0'] = c_main[0][2] - c_ref[0][2]
    # dff_out.loc[:,'RecA0'] = dff_out['CA0'] * dff_main['received_answers_cum']
    # dff_out.loc[:,'RecE0'] = dff_out['CE0'] * dff_main['EOimpl_cum']
    # dff_out.loc[:,'AltrA0'] = dff_out['CA0'] * dff_main['numAcceptedAnswers_cum']
    # dff_out.loc[:,'AltrE0'] = dff_out['CE0'] * dff_main['EditedPostsUpVotes_cum']
    dff_out.loc[:,'Tcum0'] = 0 # points are the same between reference and main path in period 0
    dff_out.loc[:,'isEditor0'] = 0 # points are the same between reference and main path in period 0
    dff_out.loc[:,'RxE0'] = dff_out.loc[:,'R0'] * dff_main.loc[:,'isEditor']
    dff_out.loc[:,'CAxE0'] = dff_out.loc[:,'CA0'] * dff_main.loc[:,'isEditor']
    dff_out.loc[:,'CExE0'] = dff_out.loc[:,'CE0'] * dff_main.loc[:,'isEditor']
    # dff_out.loc[:,'RecAxE0'] = dff_out.loc[:,'RecA0'] * dff_main.loc[:,'isEditor']
    # dff_out.loc[:,'RecExE0'] = dff_out.loc[:,'RecE0'] * dff_main.loc[:,'isEditor']
    # dff_out.loc[:,'AltrAxE0'] = dff_out.loc[:,'AltrA0'] * dff_main.loc[:,'isEditor']
    # dff_out.loc[:,'AltrExE0'] = dff_out.loc[:,'AltrE0'] * dff_main.loc[:,'isEditor']
    
    # initialize matrices for later use
    extra_main = ssp.csr_matrix( np.repeat([[0]],len(dff_main), axis=0))
    tp_main = ssp.csr_matrix( np.repeat([[1]],len(dff_main), axis=0))
    extra_ref = ssp.csr_matrix( np.repeat([[0]],len(dff_ref), axis=0))
    tp_ref = ssp.csr_matrix( np.repeat([[1]],len(dff_ref), axis=0))
    for period in range(1,rhoperiods+1):
        print('Started period',period)
        '''
        Note: there are 2 sets of state variables: 
            1) states on which is based evolution of points
            2) other states
        Since I compute the exp num of points that the user has at the beginning of period t, given period t-1 choices, in the iteration
        of period t (i.e. when the variable 'period'==t), then the states of set 1) should be updated after the computation of the exp num
        of points, BUT before the ccps.
        State variables in set 2 can be updated at any time, BUT must be before ccps and before construction of variables.
        '''
        
        ### construct state variables
        # MAIN 
        dff_main.loc[:,'quality'] = c_main[period-1][1]
        #dff_main.loc[:,'numAnswers'] = c_main[period-1][0]
        #dff_main.loc[:,'numEdits'] = c_main[period-1][2]
        dff_main.loc[:,'numedits_totalothers_accepted'] = EAE.predict(dff_main[['quality','AnswerNum','Seniority_days']]) 
        meanup_main = dff_main['lambda_up'] + c_main[period-1][0]*EUV.predict(dff_main[['quality','AnswerNum','Seniority_days','numedits_totalothers_accepted']])
        meandown_main = dff_main['lambda_down'] + c_main[period-1][0]*EDV.predict(dff_main[['quality','AnswerNum','Seniority_days','numedits_totalothers_accepted']])
        fvup_main = np.arange(sst.poisson.ppf(0.001, mu=meanup_main.min()),sst.poisson.ppf(0.999, mu=meanup_main.max())+1) # plus one to include right boundary (as it was done in simultion)
        fvdown_main = np.arange(sst.poisson.ppf(0.001, mu=meandown_main.min()),sst.poisson.ppf(0.999, mu=meandown_main.max())+1) # plus one to include right boundary (as it was done in simultion)
        accedits_main = np.arange(0,c_main[period-1][2]+1) # number of suggested edits potentially approved
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

        # # a) new points arriving
        # extra_main_new_up = ssp.csr_matrix(uppoints*fvup_main)
        # colsup = np.repeat([i for i in range(len(fvup_main))], len(fvdown_main)*len(accedits_main))
        # extra_main_new_up = extra_main_new_up[:,colsup]
        
        # extra_main_new_down = ssp.csr_matrix(downpoints*fvdown_main)
        # colsdown = np.tile(np.repeat([i for i in range(len(fvdown_main))], len(accedits_main)), (len(fvup_main)))
        # extra_main_new_down = extra_main_new_down[:,colsdown]
        
        # extra_main_new_edits = ssp.csr_matrix(approvalpoints*accedits_main)
        # colsedits = np.tile([i for i in range(len(accedits_main))], (int(extra_main_new_up.shape[1]/len(accedits_main))))
        # extra_main_new_edits = extra_main_new_edits[:,colsedits]
        
        # extra_main_new = extra_main_new_up - extra_main_new_down + extra_main_new_edits
        
        # # b) final num points arriving summing all periods after period 0
        # rows = np.repeat(0, len(dff_main), axis=0)
        # colsnew = np.repeat([i for i in range(extra_main_new.shape[1])], extra_main.shape[1])
        # colspast = np.tile([i for i in range(extra_main.shape[1])], (extra_main_new.shape[1]))

        # extra_main_new = extra_main_new[rows,:] # slice in separate commands to be sure it works fine
        # extra_main_new = extra_main_new[:,colsnew] 
        
        # extra_main = extra_main[:,colspast]
        # extra_main = extra_main + extra_main_new            
                
        # # probabilities of new points arriving / of values at a)
        # tpup_main_new =  ssp.csr_matrix(sst.poisson.pmf(fvup_main,mu=meanup_main[:,np.newaxis]))
        # tpup_main_new = tpup_main_new[:,colsup] # use column indeces created before
        
        # tpdown_main_new =  ssp.csr_matrix(sst.poisson.pmf(fvdown_main,mu=meandown_main[:,np.newaxis]))
        # tpdown_main_new = tpdown_main_new[:,colsdown]
        
        # tpedits_main = ssp.csr_matrix(sst.binom.pmf(accedits_main, c_main[period-1][2], prob_acceptance*dff_main['editIsSuggested'][:,np.newaxis]))
        # tpedits_main = tpedits_main[:,colsedits]
        
        # tp_main_new = tpup_main_new.multiply(tpdown_main_new).multiply(tpedits_main)
        
        # # joint probabilities given past / of values at b)
        # tp_main_new = tp_main_new[:,colsnew]
        # tp_main = tp_main[:,colspast]
        # tp_main = tp_main.multiply(tp_main_new)

        # # reduce dimensionality
        # fut_vals_main = extra_main[0,:].toarray()[0,:]
        # unique_fut_vals_main = np.unique(fut_vals_main)
        # tp_reduced_main = ssp.lil_matrix((extra_main.shape[0],unique_fut_vals_main.shape[0]))
        # cols_tokeep = [] # columns to keep in extra_main
        # count = 0 # to keep order of columns in new tp_main
        # for val in list(unique_fut_vals_main):
        #     colsval = np.where(fut_vals_main==val)[0]
        #     cols_tokeep.append(colsval[0]) # collects one column for each value to retain from extra_main
        #     tp_reduced_main[:,count] = tp_main[:,colsval].sum(axis=1)
        #     count += 1
        # tp_main = tp_reduced_main.tocsr()
        # extra_main = extra_main[:,cols_tokeep]
        
        #########################
        
        
        # a) new points arriving
        extra_main_new_up = ssp.csr_matrix(uppoints*fvup_main)
        colsup = np.repeat([i for i in range(len(fvup_main))], len(fvdown_main)*len(accedits_main))
        extra_main_new_up = extra_main_new_up[:,colsup]
        
        extra_main_new_down = ssp.csr_matrix(downpoints*fvdown_main)
        colsdown = np.tile(np.repeat([i for i in range(len(fvdown_main))], len(accedits_main)), (len(fvup_main)))
        extra_main_new_down = extra_main_new_down[:,colsdown]
        
        extra_main_new_edits = ssp.csr_matrix(approvalpoints*accedits_main)
        colsedits = np.tile([i for i in range(len(accedits_main))], (int(extra_main_new_up.shape[1]/len(accedits_main))))
        extra_main_new_edits = extra_main_new_edits[:,colsedits]
        
        extra_main_new = extra_main_new_up - extra_main_new_down + extra_main_new_edits
                
        # probabilities of new points arriving / of values at a)
        tpup_main_new =  ssp.csr_matrix(sst.poisson.pmf(fvup_main,mu=meanup_main[:,np.newaxis]))
        tpup_main_new = tpup_main_new[:,colsup] # use column indeces created before
        
        tpdown_main_new =  ssp.csr_matrix(sst.poisson.pmf(fvdown_main,mu=meandown_main[:,np.newaxis]))
        tpdown_main_new = tpdown_main_new[:,colsdown]
        
        tpedits_main = ssp.csr_matrix(sst.binom.pmf(accedits_main, c_main[period-1][2], prob_acceptance*dff_main['editIsSuggested'][:,np.newaxis]))
        tpedits_main = tpedits_main[:,colsedits]
        
        tp_main_new = tpup_main_new.multiply(tpdown_main_new).multiply(tpedits_main)

        # reduce dimensionality
        unique_fut_vals_main = np.unique(extra_main_new.toarray()[0], return_inverse=True)
        extra_main_new = ssp.csr_matrix(unique_fut_vals_main[0])
        tp_reduced_main = ssp.lil_matrix((tp_main_new.shape[0],unique_fut_vals_main[0].shape[0]))
        for val in range(len(unique_fut_vals_main[0])):
            colsval = np.where(unique_fut_vals_main[1]==val)[0]
            tp_reduced_main[:,val] = tp_main_new[:,colsval].sum(axis=1)
        tp_main_new = tp_reduced_main.tocsr()

        # b) final num points arriving summing all periods after period 0
        rows = np.repeat(0, len(dff_main), axis=0)
        colsnew = np.repeat([i for i in range(extra_main_new.shape[1])], extra_main.shape[1])
        colspast = np.tile([i for i in range(extra_main.shape[1])], (extra_main_new.shape[1]))

        extra_main_new = extra_main_new[rows,:] # slice in separate commands to be sure it works fine
        extra_main_new = extra_main_new[:,colsnew] 
        
        extra_main = extra_main[:,colspast]
        extra_main = extra_main + extra_main_new 

        # joint probabilities given past / of values at b)
        tp_main_new = tp_main_new[:,colsnew]
        tp_main = tp_main[:,colspast]
        tp_main = tp_main.multiply(tp_main_new)       
        
        # reduce dimensionality
        fut_vals_main = extra_main[0,:].toarray()[0,:]
        unique_fut_vals_main = np.unique(fut_vals_main)
        tp_reduced_main = ssp.lil_matrix((extra_main.shape[0],unique_fut_vals_main.shape[0]))
        cols_tokeep = [] # columns to keep in extra_main
        count = 0 # to keep order of columns in new tp_main
        for val in list(unique_fut_vals_main):
            colsval = np.where(fut_vals_main==val)[0]
            cols_tokeep.append(colsval[0]) # collects one column for each value to retain from extra_main
            tp_reduced_main[:,count] = tp_main[:,colsval].sum(axis=1)
            count += 1
        tp_main = tp_reduced_main.tocsr()
        extra_main = extra_main[:,cols_tokeep]
        
        #################################
        

        # c) final cumulated reputation obtained after all periods after period 0
        repl = np.repeat(0 ,extra_main.shape[1]) # in sparse matrices it's only possible to sum matrices of same dimension
        totextra_main = extra_main + ssp.csr_matrix(dff_main['rep_cum'][:,np.newaxis])[:,repl] # possible final rep points. HERE dff_main['rep_cum'] IS THE VALUE AT PERIOD 0
        
        # construct matrix of 1 and zeros, 1 if user is Editor
        beta_index = dff_main.loc[dff_main['isDesigned']==0].index.tolist() # needs that index are ordered numbers # why it does need????
        designed_index = dff_main.loc[dff_main['isDesigned']==1].index.tolist()
        ind_main, cols_main, dt_main = ssp.find(totextra_main)
        
        dt_main_Editor = np.where((np.isin(ind_main,designed_index) & (dt_main>=TTdesigned)) | (np.isin(ind_main,beta_index) & (dt_main>=TTbeta)),1,0)
        totextraEbin_main = ssp.csr_matrix((dt_main_Editor, (ind_main,cols_main)))
        
        # rep cum values IF above threshold
        totextraE_main = totextra_main.multiply(totextraEbin_main)
        
        # dummy - Tcum
        dt_main_tcum = np.where(np.isin(ind_main,designed_index), np.digitize(dt_main, Tgrad) , np.digitize(dt_main, Tbeta))
        Tcum_main = ssp.csr_matrix((dt_main_tcum, (ind_main,cols_main)))
                                    
        
        ### update other vaiables
        # day
        dff_main.loc[:,'datenum'] = dff_main['datenum'] + 1
        # update probability that edits are suggested
        dff_main.loc[:,'editIsSuggested'] = tp_main.multiply(1 - totextraEbin_main.toarray()).sum(axis=1)
        # update experience
        dff_main.loc[:,'AnswerNum'] = dff_main['AnswerNum'] + c_main[period-1][0]
        dff_main.loc[:,'Seniority_days'] = dff_main['Seniority_days'] + 1
        # update avail
        dff_main.loc[:,'avail'] = dff_main['avail'] + rateavail
        # update lambdas
        dff_main.loc[:,'lambda_up'] = meanup_main * np.exp(-1/tau_up)
        dff_main.loc[:,'lambda_down'] = meandown_main * np.exp(-1/tau_down)
        # period
        dff_main.loc[:,'periods'] = dff_main['periods'] + 1
        # # reciprocity: received answers
        # dff_main.loc[:,'received_answers_cum'] = dff_main['received_answers_cum'] + ERA_predict(dff_main['received_answers_cum'], dff_main['numQuestions_cum'], ERA_params) 
        # # reciprocity : received edits
        # dff_main.loc[:,'EOimpl_cum'] = dff_main['EOimpl_cum'] + c_main[period-1][0] * dff_main['numedits_totalothers_accepted']
        # # altruism variables are not updated
        
        ### variables to use for final data
        dff_main.loc[:,'R'] = tp_main.multiply(totextra_main).sum(axis=1)
        dff_main.loc[:,'CA'] = (c_main[period][0]**(maxavail / np.log(dff_main['avail'])) + c_main[period][1])
        dff_main.loc[:,'CE'] = c_main[period][2]
        # dff_main.loc[:,'RecA'] = dff_main['CA'] * dff_main['received_answers_cum']
        # dff_main.loc[:,'RecE'] = dff_main['CE'] * dff_main['EOimpl_cum']
        # dff_main.loc[:,'AltrA'] = dff_main['CA'] * dff_main['numAcceptedAnswers_cum']
        # dff_main.loc[:,'AltrE'] = dff_main['CE'] * dff_main['EditedPostsUpVotes_cum']
        dff_main.loc[:,'Tcum'] = tp_main.multiply(Tcum_main).sum(axis=1)
        dff_main.loc[:,'isEditor'] = tp_main.multiply(totextraEbin_main).sum(axis=1)
        dff_main.loc[:,'RxE'] = tp_main.multiply(totextraE_main).sum(axis=1)
        dff_main.loc[:,'CAxE'] = dff_main['CA'] * dff_main['isEditor']
        dff_main.loc[:,'CExE'] = dff_main['CE'] * dff_main['isEditor']
        # dff_main.loc[:,'RecAxE'] = dff_main['RecA'] * dff_main['isEditor']
        # dff_main.loc[:,'RecExE'] = dff_main['RecE'] * dff_main['isEditor']
        # dff_main.loc[:,'AltrAxE'] = dff_main['AltrA'] * dff_main['isEditor']
        # dff_main.loc[:,'AltrExE'] = dff_main['AltrE'] * dff_main['isEditor']        
        
        # REF
        dff_ref.loc[:,'quality'] = c_ref[period-1][1]
        #dff_ref.loc[:,'numAnswers'] = c_ref[period-1][0]
        #dff_ref.loc[:,'numEdits'] = c_ref[period-1][2]
        dff_ref.loc[:,'numedits_totalothers_accepted'] = EAE.predict(dff_ref[['quality','AnswerNum','Seniority_days']]) 
        meanup_ref = dff_ref['lambda_up'] + c_ref[period-1][0]*EUV.predict(dff_ref[['quality','AnswerNum','Seniority_days','numedits_totalothers_accepted']])
        meandown_ref = dff_ref['lambda_down'] + c_ref[period-1][0]*EDV.predict(dff_ref[['quality','AnswerNum','Seniority_days','numedits_totalothers_accepted']])
        fvup_ref = np.arange(sst.poisson.ppf(0.001, mu=meanup_ref.min()),sst.poisson.ppf(0.999, mu=meanup_ref.max())+1) # plus one to include right boundary (as it was done in simultion)
        fvdown_ref = np.arange(sst.poisson.ppf(0.001, mu=meandown_ref.min()),sst.poisson.ppf(0.999, mu=meandown_ref.max())+1) # plus one to include right boundary (as it was done in simultion)
        accedits_ref = np.arange(0,c_ref[period-1][2]+1) # number of suggested edits potentially approved
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

        # # a) new points arriving
        # extra_ref_new_up = ssp.csr_matrix(uppoints*fvup_ref)
        # colsup = np.repeat([i for i in range(len(fvup_ref))], len(fvdown_ref)*len(accedits_ref))
        # extra_ref_new_up = extra_ref_new_up[:,colsup]
        
        # extra_ref_new_down = ssp.csr_matrix(downpoints*fvdown_ref)
        # colsdown = np.tile(np.repeat([i for i in range(len(fvdown_ref))], len(accedits_ref)), (len(fvup_ref)))
        # extra_ref_new_down = extra_ref_new_down[:,colsdown]
        
        # extra_ref_new_edits = ssp.csr_matrix(approvalpoints*accedits_ref)
        # colsedits = np.tile([i for i in range(len(accedits_ref))], (int(extra_ref_new_up.shape[1]/len(accedits_ref))))
        # extra_ref_new_edits = extra_ref_new_edits[:,colsedits]
        
        # extra_ref_new = extra_ref_new_up - extra_ref_new_down + extra_ref_new_edits
        
        # # b) final num points arriving summing all periods after period 0
        # rows = np.repeat(0, len(dff_ref), axis=0)
        # colsnew = np.repeat([i for i in range(extra_ref_new.shape[1])], extra_ref.shape[1])
        # colspast = np.tile([i for i in range(extra_ref.shape[1])], (extra_ref_new.shape[1]))

        # extra_ref_new = extra_ref_new[rows,:] # slice in separate commands to be sure it works fine
        # extra_ref_new = extra_ref_new[:,colsnew] 
        
        # extra_ref = extra_ref[:,colspast]
        # extra_ref = extra_ref + extra_ref_new    
               
        # # probabilities of new points arriving / of values at a)
        # tpup_ref_new =  ssp.csr_matrix(sst.poisson.pmf(fvup_ref,mu=meanup_ref[:,np.newaxis]))
        # tpup_ref_new = tpup_ref_new[:,colsup] # use column indeces created before
        
        # tpdown_ref_new =  ssp.csr_matrix(sst.poisson.pmf(fvdown_ref,mu=meandown_ref[:,np.newaxis]))
        # tpdown_ref_new = tpdown_ref_new[:,colsdown]
        
        # tpedits_ref = ssp.csr_matrix(sst.binom.pmf(accedits_ref, c_ref[period-1][2], prob_acceptance*dff_ref['editIsSuggested'][:,np.newaxis]))
        # tpedits_ref = tpedits_ref[:,colsedits]
        
        # tp_ref_new = tpup_ref_new.multiply(tpdown_ref_new).multiply(tpedits_ref)
        
        # # joint probabilities given past / of values at b)
        # tp_ref_new = tp_ref_new[:,colsnew]
        # tp_ref = tp_ref[:,colspast]
        # tp_ref = tp_ref.multiply(tp_ref_new)

        # # reduce dimensionality
        # fut_vals_ref = extra_ref[0,:].toarray()[0,:]
        # unique_fut_vals_ref = np.unique(fut_vals_ref)
        # tp_reduced_ref = ssp.lil_matrix((extra_ref.shape[0],unique_fut_vals_ref.shape[0]))
        # cols_tokeep = [] # columns to keep in extra_ref
        # count = 0 # to keep order of columns in new tp_ref
        # for val in list(unique_fut_vals_ref):
        #     colsval = np.where(fut_vals_ref==val)[0]
        #     cols_tokeep.append(colsval[0]) # collects one column for each value to retain from extra_ref
        #     tp_reduced_ref[:,count] = tp_ref[:,colsval].sum(axis=1)
        #     count += 1
        # tp_ref = tp_reduced_ref.tocsr()
        # extra_ref = extra_ref[:,cols_tokeep]

        #########################
        
        
        # a) new points arriving
        extra_ref_new_up = ssp.csr_matrix(uppoints*fvup_ref)
        colsup = np.repeat([i for i in range(len(fvup_ref))], len(fvdown_ref)*len(accedits_ref))
        extra_ref_new_up = extra_ref_new_up[:,colsup]
        
        extra_ref_new_down = ssp.csr_matrix(downpoints*fvdown_ref)
        colsdown = np.tile(np.repeat([i for i in range(len(fvdown_ref))], len(accedits_ref)), (len(fvup_ref)))
        extra_ref_new_down = extra_ref_new_down[:,colsdown]
        
        extra_ref_new_edits = ssp.csr_matrix(approvalpoints*accedits_ref)
        colsedits = np.tile([i for i in range(len(accedits_ref))], (int(extra_ref_new_up.shape[1]/len(accedits_ref))))
        extra_ref_new_edits = extra_ref_new_edits[:,colsedits]
        
        extra_ref_new = extra_ref_new_up - extra_ref_new_down + extra_ref_new_edits
                
        # probabilities of new points arriving / of values at a)
        tpup_ref_new =  ssp.csr_matrix(sst.poisson.pmf(fvup_ref,mu=meanup_ref[:,np.newaxis]))
        tpup_ref_new = tpup_ref_new[:,colsup] # use column indeces created before
        
        tpdown_ref_new =  ssp.csr_matrix(sst.poisson.pmf(fvdown_ref,mu=meandown_ref[:,np.newaxis]))
        tpdown_ref_new = tpdown_ref_new[:,colsdown]
        
        tpedits_ref = ssp.csr_matrix(sst.binom.pmf(accedits_ref, c_ref[period-1][2], prob_acceptance*dff_ref['editIsSuggested'][:,np.newaxis]))
        tpedits_ref = tpedits_ref[:,colsedits]
        
        tp_ref_new = tpup_ref_new.multiply(tpdown_ref_new).multiply(tpedits_ref)

        # reduce dimensionality
        unique_fut_vals_ref = np.unique(extra_ref_new.toarray()[0], return_inverse=True)
        extra_ref_new = ssp.csr_matrix(unique_fut_vals_ref[0])
        tp_reduced_ref = ssp.lil_matrix((tp_ref_new.shape[0],unique_fut_vals_ref[0].shape[0]))
        for val in range(len(unique_fut_vals_ref[0])):
            colsval = np.where(unique_fut_vals_ref[1]==val)[0]
            tp_reduced_ref[:,val] = tp_ref_new[:,colsval].sum(axis=1)
        tp_ref_new = tp_reduced_ref.tocsr()

        # b) final num points arriving summing all periods after period 0
        rows = np.repeat(0, len(dff_ref), axis=0)
        colsnew = np.repeat([i for i in range(extra_ref_new.shape[1])], extra_ref.shape[1])
        colspast = np.tile([i for i in range(extra_ref.shape[1])], (extra_ref_new.shape[1]))

        extra_ref_new = extra_ref_new[rows,:] # slice in separate commands to be sure it works fine
        extra_ref_new = extra_ref_new[:,colsnew] 
        
        extra_ref = extra_ref[:,colspast]
        extra_ref = extra_ref + extra_ref_new 

        # joint probabilities given past / of values at b)
        tp_ref_new = tp_ref_new[:,colsnew]
        tp_ref = tp_ref[:,colspast]
        tp_ref = tp_ref.multiply(tp_ref_new)       
        
        # reduce dimensionality
        fut_vals_ref = extra_ref[0,:].toarray()[0,:]
        unique_fut_vals_ref = np.unique(fut_vals_ref)
        tp_reduced_ref = ssp.lil_matrix((extra_ref.shape[0],unique_fut_vals_ref.shape[0]))
        cols_tokeep = [] # columns to keep in extra_ref
        count = 0 # to keep order of columns in new tp_ref
        for val in list(unique_fut_vals_ref):
            colsval = np.where(fut_vals_ref==val)[0]
            cols_tokeep.append(colsval[0]) # collects one column for each value to retain from extra_ref
            tp_reduced_ref[:,count] = tp_ref[:,colsval].sum(axis=1)
            count += 1
        tp_ref = tp_reduced_ref.tocsr()
        extra_ref = extra_ref[:,cols_tokeep]
        
        #################################

        # c) final cumulated reputation obtained after all periods after period 0
        repl = np.repeat(0 ,extra_ref.shape[1]) # in sparse matrices it's only possible to sum matrices of same dimension
        totextra_ref = extra_ref + ssp.csr_matrix(dff_ref['rep_cum'][:,np.newaxis])[:,repl] # possible final rep points. HERE dff_ref['rep_cum'] IS THE VALUE AT PERIOD 0
        
        # construct matrix of 1 and zeros, 1 if user is Editor
        beta_index = dff_ref.loc[dff_ref['isDesigned']==0].index.tolist() # needs that index are ordered numbers # why it does need????
        designed_index = dff_ref.loc[dff_ref['isDesigned']==1].index.tolist()
        ind_ref, cols_ref, dt_ref = ssp.find(totextra_ref)
        
        dt_ref_Editor = np.where((np.isin(ind_ref,designed_index) & (dt_ref>=TTdesigned)) | (np.isin(ind_ref,beta_index) & (dt_ref>=TTbeta)),1,0)
        totextraEbin_ref = ssp.csr_matrix((dt_ref_Editor, (ind_ref,cols_ref)))
        
        # rep cum values IF above threshold
        totextraE_ref = totextra_ref.multiply(totextraEbin_ref)
        
        # dummy - Tcum
        dt_ref_tcum = np.where(np.isin(ind_ref,designed_index), np.digitize(dt_ref, Tgrad) , np.digitize(dt_ref, Tbeta))
        Tcum_ref = ssp.csr_matrix((dt_ref_tcum, (ind_ref,cols_ref)))
        
        ### update vaiables
        # day
        dff_ref.loc[:,'datenum'] = dff_ref['datenum'] + 1
        # update probability that edits are suggested
        dff_ref.loc[:,'editIsSuggested'] = tp_ref.multiply(1 - totextraEbin_ref.toarray()).sum(axis=1)
        # update experience
        dff_ref.loc[:,'AnswerNum'] = dff_ref['AnswerNum'] + c_ref[period-1][0]
        dff_ref.loc[:,'Seniority_days'] = dff_ref['Seniority_days'] + 1
        # update avail
        dff_ref.loc[:,'avail'] = dff_ref['avail'] + rateavail
        # update lambdas
        dff_ref.loc[:,'lambda_up'] = meanup_ref * np.exp(-1/tau_up)
        dff_ref.loc[:,'lambda_down'] = meandown_ref * np.exp(-1/tau_down)
        # period
        dff_ref.loc[:,'periods'] = dff_ref['periods'] + 1
        # # reciprocity: received answers
        # dff_ref.loc[:,'received_answers_cum'] = dff_ref['received_answers_cum'] + ERA_predict(dff_ref['received_answers_cum'], dff_ref['numQuestions_cum'], ERA_params) 
        # # reciprocity : received edits
        # dff_ref.loc[:,'EOimpl_cum'] = dff_ref['EOimpl_cum'] + c_ref[period-1][0] * dff_ref['numedits_totalothers_accepted']
        # # altruism variables are not updated
        
        ### variables to use for final data
        dff_ref.loc[:,'R'] = tp_ref.multiply(totextra_ref).sum(axis=1)
        dff_ref.loc[:,'CA'] = (c_ref[period][0]**(maxavail / np.log(dff_ref['avail'])) + c_ref[period][1])
        dff_ref.loc[:,'CE'] = c_ref[period][2]
        # dff_ref.loc[:,'RecA'] = dff_ref['CA'] * dff_ref['received_answers_cum']
        # dff_ref.loc[:,'RecE'] = dff_ref['CE'] * dff_ref['EOimpl_cum']
        # dff_ref.loc[:,'AltrA'] = dff_ref['CA'] * dff_ref['numAcceptedAnswers_cum']
        # dff_ref.loc[:,'AltrE'] = dff_ref['CE'] * dff_ref['EditedPostsUpVotes_cum']
        dff_ref.loc[:,'Tcum'] = tp_ref.multiply(Tcum_ref).sum(axis=1)
        dff_ref.loc[:,'isEditor'] = tp_ref.multiply(totextraEbin_ref).sum(axis=1)
        dff_ref.loc[:,'RxE'] = tp_ref.multiply(totextraE_ref).sum(axis=1)
        dff_ref.loc[:,'CAxE'] = dff_ref['CA'] * dff_ref['isEditor']
        dff_ref.loc[:,'CExE'] = dff_ref['CE'] * dff_ref['isEditor']
        # dff_ref.loc[:,'RecAxE'] = dff_ref['RecA'] * dff_ref['isEditor']
        # dff_ref.loc[:,'RecExE'] = dff_ref['RecE'] * dff_ref['isEditor']
        # dff_ref.loc[:,'AltrAxE'] = dff_ref['AltrA'] * dff_ref['isEditor']
        # dff_ref.loc[:,'AltrExE'] = dff_ref['AltrE'] * dff_ref['isEditor']        
        
        ### OUT VARIABLES
        dff_out.loc[:,'R%d'%(period)] = dff_main['R'] - dff_ref['R']
        dff_out.loc[:,'CA%d'%(period)] = dff_main['CA'] - dff_ref['CA']
        dff_out.loc[:,'CE%d'%(period)] = dff_main['CE'] - dff_ref['CE']
        # dff_out.loc[:,'RecA%d'%(period)] = dff_main['RecA'] - dff_ref['RecA']
        # dff_out.loc[:,'RecE%d'%(period)] = dff_main['RecE'] - dff_ref['RecE']
        # dff_out.loc[:,'AltrA%d'%(period)] = dff_main['AltrA'] - dff_ref['AltrA']
        # dff_out.loc[:,'AltrE%d'%(period)] = dff_main['AltrE'] - dff_ref['AltrE']
        dff_out.loc[:,'Tcum%d'%(period)] = dff_main['Tcum'] - dff_ref['Tcum']
        dff_out.loc[:,'isEditor%d'%(period)] = dff_main['isEditor'] - dff_ref['isEditor']
        dff_out.loc[:,'RxE%d'%(period)] = dff_main['RxE'] - dff_ref['RxE']
        dff_out.loc[:,'CAxE%d'%(period)] = dff_main['CAxE'] - dff_ref['CAxE']
        dff_out.loc[:,'CExE%d'%(period)] = dff_main['CExE'] - dff_ref['CExE']
        # dff_out.loc[:,'RecAxE%d'%(period)] = dff_main['RecAxE'] - dff_ref['RecAxE']
        # dff_out.loc[:,'RecExE%d'%(period)] = dff_main['RecExE'] - dff_ref['RecExE']
        # dff_out.loc[:,'AltrAxE%d'%(period)] = dff_main['AltrAxE'] - dff_ref['AltrAxE']
        # dff_out.loc[:,'AltrExE%d'%(period)] = dff_main['AltrExE'] - dff_ref['AltrExE']        
        
        ## ccps
        
        # MAIN
        # predicted probabilities
        ccps_main = ssp.lil_matrix((len(dff_main),totextra_main.shape[1])) # compute ccps fo each possible future states and add to sparse matrix, then take expectation
        for i in range(totextra_main.shape[1]):
            futdata = np.array([totextra_main[:,i].toarray().flatten(), dff_main['lambda_up'],dff_main['lambda_down'],dff_main['avail'],dff_main['AnswerNum'],dff_main['Seniority_days'],dff_main['periods'],
                                dff_main['datenum'], dff_main['Tcum']]).T
            futdata = scaler.transform(futdata)
            p = clf.predict_log_proba(futdata)
            p = pd.DataFrame(p, columns=clf.classes_)
            p = p.loc[:,choice_tupl2num[c_main[period]]].values
            ccps_main[:,i] = p[:,np.newaxis]
        ccps_main = ccps_main.tocsr()
        dff_main.loc[:,'ccp'] = tp_main.multiply(ccps_main).sum(axis=1) # expected value of ccp
        

        # REF
        # predicted probabilities
        ccps_ref = ssp.lil_matrix((len(dff_ref),totextra_ref.shape[1])) # compute ccps fo each possible future states and add to sparse matrix, then take expectation
        for i in range(totextra_ref.shape[1]):
            futdata = np.array([totextra_ref[:,i].toarray().flatten(), dff_ref['lambda_up'],dff_ref['lambda_down'],dff_ref['avail'],dff_ref['AnswerNum'],dff_ref['Seniority_days'],dff_ref['periods'],
                                dff_ref['datenum'], dff_ref['Tcum']]).T
            futdata = scaler.transform(futdata)
            p = clf.predict_log_proba(futdata)
            p = pd.DataFrame(p, columns=clf.classes_)
            p = p.loc[:,choice_tupl2num[c_ref[period]]].values
            ccps_ref[:,i] = p[:,np.newaxis]
        ccps_ref = ccps_ref.tocsr()
        dff_ref.loc[:,'ccp'] = tp_ref.multiply(ccps_ref).sum(axis=1) # expected value of ccp
        
        
        dff_out.loc[:,'ccp%d'%(period)] = dff_main['ccp'] - dff_ref['ccp']
        
    dff_out.to_csv(out_dir + 'states_%s_wsparse_week.csv'%(choicestr))

### load all dfs constructed and sort original data
alldfs = {}
for choice in choices:
    choicestr = '%f_%f_%f'%choice
    datachoice = pd.read_csv(out_dir + 'states_%s_wsparse_week.csv'%(choicestr), index_col=0)
    datachoice = datachoice.sort_values(by=['user','periods'])
    alldfs[choice] = datachoice
hist = hist.sort_values(by=['user','periods'])

## str to num
#choice_str2num = {}
#count =0
#for choice in choices:
#    choicestr = '%f_%f_%f'%choice
#    choice_str2num[choicestr] = count
#    count += 1

# # temp scale vars (must be done before constructing the data)
# for choice in choices:
#     alldfs[choice].loc[:,'AltrA0'] =  alldfs[choice]['AltrA0'] / 100000
#     alldfs[choice].loc[:,'AltrE0'] =  alldfs[choice]['AltrE0'] / 100000
#     alldfs[choice].loc[:,'AltrAxE0'] =  alldfs[choice]['AltrAxE0'] / 100000
#     alldfs[choice].loc[:,'AltrExE0'] =  alldfs[choice]['AltrExE0'] / 100000
#     if 'AltrA1' in  alldfs[choice].columns:
#         alldfs[choice].loc[:,'AltrA1'] =  alldfs[choice]['AltrA1'] / 100000
#         alldfs[choice].loc[:,'AltrE1'] =  alldfs[choice]['AltrE1'] / 100000
#         alldfs[choice].loc[:,'AltrAxE1'] =  alldfs[choice]['AltrAxE1'] / 100000
#         alldfs[choice].loc[:,'AltrExE1'] =  alldfs[choice]['AltrExE1'] / 100000
    

    
# first construct value function evaluation for each possible choice
def estim(params, truedata, inputdata):
    diff_cvfs = pd.DataFrame()
    for choice in choices:
        param_vec = params
        #choicestr = '%f_%f_%f'%choice
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
    
    # x = np.abs(diff_cvfs).max().max()
    # if x > 700:
    #     print('adgustment of support')
    #     if diff_cvfs.max().max() > 700:
    #         diff_cvfs = diff_cvfs - (x - 700)
    #     else:
    #         diff_cvfs = diff_cvfs + (x - 700)
    
    den = np.log(np.sum(np.exp(diff_cvfs), axis=1))
    diff_cvfs['truechoice'] = truedata['choicenum'].values
    diff_cvfs.set_index('truechoice', append=True, inplace=True)
    num = diff_cvfs.stack().reset_index(level=[-1,-2])
    num.rename(columns={'level_2':'choice',0:'num'}, inplace=True)
    #num.loc[:,'choice'] = num['choice'].apply(lambda x: choice_str2num[x])
    #num.loc[:,'truechoice'] = num['truechoice'].apply(lambda x: choice_tupl2num[x])
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
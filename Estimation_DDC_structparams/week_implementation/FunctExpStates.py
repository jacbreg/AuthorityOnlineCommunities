#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 16:12:13 2020

@author: jacopo
"""

'''
FIRST STEP OF ESTIMATION OF FLOW PAYOFF PARAMETERS

function to construct expected value of state variables under finite dependence
'''

import pandas as pd
import numpy as np
import scipy.sparse as ssp
import scipy.stats as sst
import math

def ExpStates(choices, # choice set
              hist, # data
              relevant_columns, # columns necessary for computations
              maxanswernum,
              maxseniority,
              maxavail,
              EAE,
              EUV,
              EDV,
              tau_up,
              tau_down,
              rateavail,
              prob_acceptance,
              uppoints,
              downpoints,
              approvalpoints,
              Tgrad,
              Tbeta,
              TTdesigned,
              TTbeta,
              clf,
              scaler,
              choice_tupl2num,
              typenum,
              out_dir):
    for choice in choices:
        
        print('started choice',choice)
        choicestr = '%f_%f_%f'%choice
        
        # df to store final variables
        dff_out = hist[['user','periods']]
        
        # df to store state values for each choice path
        dff_main = hist[relevant_columns].copy()
        dff_ref = hist[relevant_columns].copy()
    
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
        dff_out.loc[:,'Tcum0'] = 0 # points are the same between reference and main path in period 0
        dff_out.loc[:,'isEditor0'] = 0 # points are the same between reference and main path in period 0
        dff_out.loc[:,'RxE0'] = dff_out.loc[:,'R0'] * dff_main.loc[:,'isEditor']
        dff_out.loc[:,'CAxE0'] = dff_out.loc[:,'CA0'] * dff_main.loc[:,'isEditor']
        dff_out.loc[:,'CExE0'] = dff_out.loc[:,'CE0'] * dff_main.loc[:,'isEditor']
    
        
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
    
            
            ### variables to use for final data
            dff_main.loc[:,'R'] = tp_main.multiply(totextra_main).sum(axis=1)
            dff_main.loc[:,'CA'] = (c_main[period][0]**(maxavail / np.log(dff_main['avail'])) + c_main[period][1])
            dff_main.loc[:,'CE'] = c_main[period][2]
            dff_main.loc[:,'Tcum'] = tp_main.multiply(Tcum_main).sum(axis=1)
            dff_main.loc[:,'isEditor'] = tp_main.multiply(totextraEbin_main).sum(axis=1)
            dff_main.loc[:,'RxE'] = tp_main.multiply(totextraE_main).sum(axis=1)
            dff_main.loc[:,'CAxE'] = dff_main['CA'] * dff_main['isEditor']
            dff_main.loc[:,'CExE'] = dff_main['CE'] * dff_main['isEditor']     
            
            # REF
            dff_ref.loc[:,'quality'] = c_ref[period-1][1]
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
            
            ### variables to use for final data
            dff_ref.loc[:,'R'] = tp_ref.multiply(totextra_ref).sum(axis=1)
            dff_ref.loc[:,'CA'] = (c_ref[period][0]**(maxavail / np.log(dff_ref['avail'])) + c_ref[period][1])
            dff_ref.loc[:,'CE'] = c_ref[period][2]
            dff_ref.loc[:,'Tcum'] = tp_ref.multiply(Tcum_ref).sum(axis=1)
            dff_ref.loc[:,'isEditor'] = tp_ref.multiply(totextraEbin_ref).sum(axis=1)
            dff_ref.loc[:,'RxE'] = tp_ref.multiply(totextraE_ref).sum(axis=1)
            dff_ref.loc[:,'CAxE'] = dff_ref['CA'] * dff_ref['isEditor']
            dff_ref.loc[:,'CExE'] = dff_ref['CE'] * dff_ref['isEditor']
          
            ### OUT VARIABLES
            dff_out.loc[:,'R%d'%(period)] = dff_main['R'] - dff_ref['R']
            dff_out.loc[:,'CA%d'%(period)] = dff_main['CA'] - dff_ref['CA']
            dff_out.loc[:,'CE%d'%(period)] = dff_main['CE'] - dff_ref['CE']
            dff_out.loc[:,'Tcum%d'%(period)] = dff_main['Tcum'] - dff_ref['Tcum']
            dff_out.loc[:,'isEditor%d'%(period)] = dff_main['isEditor'] - dff_ref['isEditor']
            dff_out.loc[:,'RxE%d'%(period)] = dff_main['RxE'] - dff_ref['RxE']
            dff_out.loc[:,'CAxE%d'%(period)] = dff_main['CAxE'] - dff_ref['CAxE']
            dff_out.loc[:,'CExE%d'%(period)] = dff_main['CExE'] - dff_ref['CExE']      
            
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
            
        dff_out.to_csv(out_dir + 'states_{}_wsparse_week_type{}.csv'.format(choicestr,typenum))
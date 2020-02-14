#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 11:21:53 2020

@author: jacopo
"""

'''
class code for Dynamic discrete choice estimation with finite dependendence and 
no observed characteristics included

-- version 2 has slightly more optimised routine + ccps are time dependent
'''

import numpy as np
import pandas as pd
import scipy.optimize as sopt
import scipy.sparse as ssp
import math

class DDC():
    def __init__(self, bins2interval_choices, bins2interval_states, choices, allstates_df, 
                 ccp, directory_mat, zeta, p_acceptance, delta, VARS, has_editor_dummy, truedata):
        '''VARS is a list of covariate for which it is desired coefficient estimate
        alone and interacted with Editor dummy (=1 if user has >2000 rep)'''
        if not isinstance(has_editor_dummy, bool):
            raise ValueError('has_editor_dummy has to be boolean')
        
        self.bins2interval_choices = bins2interval_choices
        self.bins2interval_states = bins2interval_states
        self.choices = choices
        self.allstates_df = allstates_df
        colnames_allstatesdf = allstates_df.columns.tolist()
        colnames_allstatesdf.remove('Editor')
        self.colnames_allstatesdf = colnames_allstatesdf
        self.ccp = ccp # now this is a logistic regression with the period as one of the variables
        self.zeta = zeta
        self.p_acceptance = p_acceptance
        self.delta = delta
        self.VARS = VARS
        self.has_editor_dummy = has_editor_dummy
        self.truedata = truedata
        if self.has_editor_dummy == True:
            self.num_covariates = len(self.VARS) * 2 + 1 # (includes interactions with privilege/Editor dummy + constant per editing dummy)
        else:
            self.num_covariates = len(self.VARS) * 2 # (includes interactions with privilege/Editor dummy)
        self.directory_mat = directory_mat

        # choice probabilities   
        c = self.ccp.predict_log_proba(allstates_df[self.colnames_allstatesdf].values)
        self.ccp_values = pd.DataFrame(c, columns=self.ccp.classes_)
        
        # matrix to fill ccvs for each possible state-action
        state_indexes = allstates_df.reset_index()
        unique_states = truedata[colnames_allstatesdf + ['Editor']].drop_duplicates()
        self.unique_states = unique_states
        rowindex = pd.merge(unique_states, state_indexes, on=colnames_allstatesdf+ ['Editor'], how='left')
        rowindex = rowindex['index'].tolist()
        self.rowindex = rowindex
        
        index = pd.MultiIndex.from_frame(self.unique_states[self.colnames_allstatesdf])
        columns = pd.MultiIndex.from_tuples(self.choices, names=['A_binsN','E_binsN'])
        self.emtpy_choice2CVF = pd.DataFrame(index=index, columns=columns)

    def computeFlow(self, st, ch, params):
        '''
        st: vector with all possible states combinations (i.e. uniquestates_df)
        ch: tuple with choices: (choice_A, choice_E)
        '''
        '''
        Other variables that could be included:
            - being editor per se
            - number of privileges obtained in t
            - total number of privileges obtained up to period t
            - number of answers accepted
            - individual characteristics
            - whether i'm a user that has reached the threshold (find a way to
                control for user that would reach it but they are not enough time
                in the platform.)
        '''
                
        params_mat = params.reshape((1,self.num_covariates)) # it fills rows by rows
        # rows are different fixed states, columns are vriables of interest
        
        possible_vars = {'points':st['rep_cum_binsN'],
                         'Achoice': np.repeat(ch[0],len(st)),
                         'Echoice': np.repeat(ch[1],len(st)),
                         'RecA': st['received_answersLAG_binsN'] * np.repeat(ch[0],len(st)),
                         'RecE': st['TotEOLAG_binsN'] * np.repeat(ch[1],len(st)),
                         'altruism':st['OtherNetUVLAG_binsN']
                         }
        
        variables = [possible_vars[i] for i in self.VARS]
                            
        #m1 = np.vstack((st['rep_cum_binsN'],Achoice,Echoice,RecA,RecE,st['OtherNetUVLAG_binsN']))
        m1 = np.vstack(variables)
        
        m2 = np.multiply(m1,st['Editor'][np.newaxis,:]) # need to transform st['Editor'] from a vector to a matrix of dim. 1 X n 
     
        m = np.vstack((m1,m2))
        
        if self.has_editor_dummy == True:
            m = np.vstack((m,st['Editor'][np.newaxis,:]))

        u = np.matmul(params_mat, m)[0]
        
        return u


    def cvf(self, params):
        #print('start')
        #print(params)
        ## first compute conditional value functions for all possible initial states / choices  
        choice2CVF = self.emtpy_choice2CVF.copy()
        #t0 = tm.time()
        for choice in self.choices: # for each initial choice
    
            if choice == (0,0): # choice zero is the same as reference so it cancels out
                choice2CVF.loc[:,choice] = 0
                continue
    
            if choice[0] == 0:
                rhoperiods = 1
            else:
                choice_val = math.ceil(self.bins2interval_choices['A'][choice[0]].mid)
                rhoperiods = round((np.log(0.01) - np.log(choice_val))/np.log(self.zeta)) + 1 # this solves for x: zeta**x * Lambda = 0.1
            
            # PERIOD 0
            u_choice = self.computeFlow(self.allstates_df, choice, params)
        
            u_zero = self.computeFlow(self.allstates_df, (0,0), params)
            
            choice_str = str(choice[0]) + '_' + str(choice[1])
            transprob_choice = ssp.load_npz(self.directory_mat + '%s.npz'%(choice_str))
            transprob_zero = ssp.load_npz(self.directory_mat + '0_0.npz')
    
            vf = u_choice[self.rowindex] - u_zero[self.rowindex]
            
            # PERIODS > 0
            ccpv_zero = self.ccp_values['0_0'].values
            if choice_str in self.ccp_values.columns:
                ccpv_choice = self.ccp_values[choice_str].values
            else:
                ccpv_choice = np.repeat(0,len(self.allstates_df))
            
            x_choice = u_choice - ccpv_choice + np.euler_gamma
            x_zero = u_zero - ccpv_zero + np.euler_gamma
    
            for period in range(1,int(rhoperiods)+1):
                
                if period == 1:
                    tp_choice = transprob_choice[self.rowindex]
                    tp_zero = transprob_zero[self.rowindex]
                    v = tp_choice.dot(x_zero) * self.delta**period
                    v_ref = tp_zero.dot(x_choice) * self.delta**period
                    
                    # create future transition probabilities
                    transprob = tp_choice.dot(transprob_zero)
                    transprob_ref = tp_zero.dot(transprob_choice)                
                    
                else:
                    v = transprob.dot(x_zero) * self.delta**period
                    v_ref = transprob_ref.dot(x_zero) * self.delta**period                
    
                    # update transition probabilities
                    transprob = transprob.dot(transprob_zero)
                    transprob_ref = transprob_ref.dot(transprob_zero)             
    
                vf = vf + v - v_ref
                
            # add to dataframe
            choice2CVF.loc[:,choice] = vf
        
        #t1 = tm.time()
        #print(t1-t0)
        return choice2CVF
    
    def negll(self, params):
        choice2CVF = self.cvf(params)
        #choice2CVF = choice2CVF.astype(np.longdouble)
        x = np.abs(choice2CVF).max().max()
        if x > 700:
            print('adgustment of support')
            if choice2CVF.max().max() > 700:
                choice2CVF = choice2CVF - (x - 700)
            else:
                choice2CVF = choice2CVF + (x - 700)
        
        # create denominator and numerator and merge to real data
        choice2CVF['den'] = sum(np.exp(choice2CVF.values).T)
        choice2CVF.set_index('den', append=True, inplace=True)
        #a = tm.time()
    
        choice2CVF = choice2CVF.stack(level=['A_binsN','E_binsN'])
        choice2CVF = choice2CVF.rename('num')
        choice2CVF = pd.DataFrame(choice2CVF)
        choice2CVF.reset_index(level='den', inplace=True)
        
        dfout = pd.merge(self.truedata, choice2CVF, on=self.colnames_allstatesdf+['A_binsN','E_binsN'],
                         how='left')

        outval = - (dfout['num'] - np.log(dfout['den'])).sum()
        print('f(x)=',outval)
        if not np.isfinite(outval):
            raise ValueError('evaluation returned infine num')
        return outval
    
    def fit(self, init_guess, method, options={'disp':True, 'maxiter':10000}):
        if len(init_guess) != self.num_covariates:
            raise ValueError('number of parameter values different from number of variables')
        return sopt.minimize(self.negll, x0=init_guess, method=method,
                             options=options)
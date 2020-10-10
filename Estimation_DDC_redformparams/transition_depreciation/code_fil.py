# -*- coding: utf-8 -*-
"""
Created on 

@author: filippo

code to estimate returns of points in future periods from today's effort

"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt 
import statsmodels.formula.api as smf
import seaborn as sns
import statsmodels.formula.api as smf

#qa_name = 'apple/'
qa_name = 'ell/'

##########################################################################

directory = '/home/jacopo/OneDrive/Dati_Jac_locali/stack/' + qa_name

data = pd.read_csv(directory + 'postHistV2.csv', dtype={'OwnerUserId':str}, parse_dates=['day'])

#########
# DAILY #
#########

# different models for upvotes and downvotes

# 1) upvotes
data.loc[:,'up_lag'] = data.groupby('PostId')['numUpvotes'].transform(lambda x: x.shift(1))

meanXt = data.groupby('periods')['numUpvotes'].mean()

# model 1: exponential (pendulum)
def expo(t, A, tau): # exponential decay function; A = aplitude, tau = decay time
	return A * np.exp(-t/tau) 

par_up, cov_up = curve_fit(expo, data['periods'].values, data['numUpvotes'].values) # fit all data together with scipy curve_fit
pd.to_pickle(par_up,directory + 'decay_params_up.pkl') # par[0]==estiamte of A, par[1]=estiamate of tau
'''
points_t = par[0] * np.exp(-t/par[1]) with the first period being t==0
'''

# model 2: AR(1) process
results = smf.ols('numUpvotes ~ up_lag -1', data=data).fit()
print(results.summary())
beta_up = results.params[0]
pd.to_pickle(beta_up,directory + 'decay_params_ar1_up.pkl')

# model r: potential
def pot(t, A, k):
    return A*(t+1)**k
par_pot_up, cov_pot_up = curve_fit(pot, data['periods'].values, data['numUpvotes'].values)
pd.to_pickle(par_pot_up,directory + 'decay_params_pot_up.pkl')

# plot two models for 10 first periods

def ar1(t, A, beta):
    return A * (beta**(t))

num_periods_toplot = 10    

means = meanXt[:num_periods_toplot+1].values
t_means = np.arange(0,num_periods_toplot+1,step=1)
t_function = np.linspace(0, num_periods_toplot+1, 60)

plt.figure()
#plt.plot(t, x, 'b.', label='data')
plt.plot(t_means, means, 'ro', label='Data means')
plt.plot(t_function, expo(t_function, *par_up), color='black',linestyle='solid', label=r'Exponential: up-votes = $\hat{A}e^{-\frac{t}{\hat{\tau}}} $')
plt.plot(t_function, ar1(t_function, means[0], beta_up),color='black',linestyle='dashed', label=r'AR(1):  up-votes = $\hat{A} \hat{\gamma}^t $')
plt.plot(t_function, pot(t_function, *par_pot_up),linestyle='dotted',color='black', label=r'Power: up-votes = $\hat{A}(t+1)^{\hat{\delta}} $')
plt.xlabel(r'days (t)') # x-label
plt.ylabel(r'Num. Upvotes') # y-label
plt.grid(True, which='both') # create background grid
plt.legend(loc=0, fontsize='xx-large') # put legend in authomatically chosen optimal location
plt.annotate("", xy=(-0.1, 0), xytext=(-0.1, means[0]),
             arrowprops=dict(arrowstyle="<->", connectionstyle="arc3"))
plt.text(-0.1, means[0]/2, r'$\hat{A}$',
         {'color': 'black', 'fontsize': 24, 'ha': 'center', 'va': 'center',
          'bbox': dict(boxstyle="round", fc="white", ec="black", pad=0.2)})
plt.tight_layout() # make a compact figure
# saved as Decay_upvotes.png


# 2) downvotes
data.loc[:,'down_lag'] = data.groupby('PostId')['numDownvotes'].transform(lambda x: x.shift(1))

meanXtd = data.groupby('periods')['numDownvotes'].mean()

# model 1: exponential (pendulum)
def expo(t, A, tau): # exponential decay function; A = aplitude, tau = decay time
	return A * np.exp(-t/tau) 

par_down, cov_down = curve_fit(expo, data['periods'].values, data['numDownvotes'].values) # fit all data together with scipy curve_fit
pd.to_pickle(par_down,directory + 'decay_params_down.pkl') # par[0]==estiamte of A, par[1]=estiamate of tau
'''
points_t = par[0] * np.exp(-t/par[1]) with the first period being t==0
'''

# model 2: AR(1) process
results = smf.ols('numDownvotes ~ down_lag -1', data=data).fit()
print(results.summary())
beta_down = results.params[0]
pd.to_pickle(beta_down,directory + 'decay_params_ar1_down.pkl')

# model r: potential
def pot(t, A, k):
    return A*(t+1)**k
par_pot_down, cov_pot_down = curve_fit(pot, data['periods'].values, data['numDownvotes'].values)
pd.to_pickle(par_pot_down,directory + 'decay_params_pot_down.pkl')

# plot two models for 10 first periods

def ar1(t, A, beta):
    return A * (beta**(t))

num_periods_toplot = 10    

means = meanXtd[:num_periods_toplot+1].values
t_means = np.arange(0,num_periods_toplot+1,step=1)
t_function = np.linspace(0, num_periods_toplot+1, 60)

plt.figure()
#plt.plot(t, x, 'b.', label='data')
plt.plot(t_means, means, 'ro', label='Data means')
plt.plot(t_function, expo(t_function, *par_down), color='black',linestyle='solid', label=r'Exponential: down-votes = $\hat{A}e^{-\frac{t}{\hat{\tau}}} $')
plt.plot(t_function, ar1(t_function, means[0], beta_down),color='black',linestyle='dashed', label=r'AR(1): down-votes = $\hat{A} \hat{\gamma}^t $')
plt.plot(t_function, pot(t_function, *par_pot_down),linestyle='dotted',color='black', label=r'Power: down-votes = $\hat{A}(t+1)^{\hat{\delta}} $')
plt.xlabel(r'days (t)') # x-label
plt.ylabel(r'Num. Downvotes') # y-label
plt.grid(True, which='both') # create background grid
plt.legend(loc=0, fontsize='xx-large') # put legend in authomatically chosen optimal location
plt.annotate("", xy=(-0.1, 0), xytext=(-0.1, means[0]),
             arrowprops=dict(arrowstyle="<->", connectionstyle="arc3"))
plt.text(-0.1, means[0]/2, r'$\hat{A}$',
         {'color': 'black', 'fontsize': 24, 'ha': 'center', 'va': 'center',
          'bbox': dict(boxstyle="round", fc="white", ec="black", pad=0.2)})
plt.tight_layout() # make a compact figure

# saved as Decay_downvotes.png



## ARE PARAMETERS DIFFERENT FOR DIFFERENT LEVELS OF EFFORT?
# create quality variable
data['points'] = (10*data['numUpvotes']) - (2*data['numDownvotes'])
data['precisionF2'] = data['precisionF']**2
data['lengthF2'] = data['lengthF']**2
data['numpictF2'] = data['numpictF']**2
data['numlinksF2'] = data['numlinksF']**2
modfit = smf.ols('points ~ precisionF + precisionF2 + lengthF + lengthF2 + numpictF + numpictF2 + numlinksF + numlinksF2',
        data=data).fit()
data['quality'] = modfit.predict(data[['precisionF','precisionF2','lengthF','lengthF2','numpictF','numpictF2','numlinksF','numlinksF2']])

subsampleA = data.loc[data['quality']<data['quality'].quantile(0.33),] # take subsample of up-votes within a selected quantile
subsampleB = data.loc[(data['quality']>=data['quality'].quantile(0.33)) & (data['quality']<data['quality'].quantile(0.66))  ,]
subsampleC = data.loc[(data['quality']>=data['quality'].quantile(0.66)),]

meanXtA = subsampleA.groupby('periods')['numUpvotes'].mean()
meanXtB = subsampleB.groupby('periods')['numUpvotes'].mean()
meanXtC = subsampleC.groupby('periods')['numUpvotes'].mean()

parA, covA = curve_fit(expo, subsampleA['periods'].values, subsampleA['numUpvotes'].values) # fit data A
parB, covB = curve_fit(expo, subsampleB['periods'].values, subsampleB['numUpvotes'].values) # fit data B
parC, covC = curve_fit(expo, subsampleC['periods'].values, subsampleC['numUpvotes'].values) # fit data C

meansA = meanXtA[:num_periods_toplot+1].values
meansB = meanXtB[:num_periods_toplot+1].values
meansC = meanXtC[:num_periods_toplot+1].values

plt.figure()
#plt.plot(t, x, 'b.', label='data')
plt.plot(t_means, meansA, 'r+', label='mean low quality')
plt.plot(t_means, meansB, 'rx', label='mean medium quality')
plt.plot(t_means, meansC, 'ro', label='mean high quality')
plt.plot(t_function, expo(t_function, *parA), label='low quality')
plt.plot(t_function, expo(t_function, *parB), label='medium quality')
plt.plot(t_function, expo(t_function, *parC), label='high quality')
plt.xlabel(r'days') # x-label
plt.ylabel(r'Num Upvotes') # y-label
plt.grid(True, which='both') # create background grid
plt.legend(loc=0) # put legend in authomatically chosen optimal location
plt.tight_layout() # make a compact figure

# saved as Decay_upvotes_byquality.png

parD, covD = curve_fit(expo, subsampleA['periods'].values, subsampleA['numDownvotes'].values) # fit data A
parE, covE = curve_fit(expo, subsampleB['periods'].values, subsampleB['numDownvotes'].values) # fit data B
parF, covF = curve_fit(expo, subsampleC['periods'].values, subsampleC['numDownvotes'].values) # fit data C

meanXtA = subsampleA.groupby('periods')['numDownvotes'].mean()
meanXtB = subsampleB.groupby('periods')['numDownvotes'].mean()
meanXtC = subsampleC.groupby('periods')['numDownvotes'].mean()

meansA = meanXtA[:num_periods_toplot+1].values
meansB = meanXtB[:num_periods_toplot+1].values
meansC = meanXtC[:num_periods_toplot+1].values

plt.figure()
#plt.plot(t, x, 'b.', label='data')
plt.plot(t_means, meansA, 'r+', label='mean low quality')
plt.plot(t_means, meansB, 'rx', label='mean medium quality')
plt.plot(t_means, meansC, 'ro', label='mean high quality')
plt.plot(t_function, expo(t_function, *parD), label='low quality')
plt.plot(t_function, expo(t_function, *parE), label='medium quality')
plt.plot(t_function, expo(t_function, *parF), label='high quality')
plt.xlabel(r'days') # x-label
plt.ylabel(r'Num Downvotes') # y-label
plt.grid(True, which='both') # create background grid
plt.legend(loc=0) # put legend in authomatically chosen optimal location
plt.tight_layout() # make a compact figure

# saved as Decay_downvotes_byquality.png

##########
# WEEKLY #
##########

aggdict = {'periods':'first',
           'numUpvotes':'sum',
           'numDownvotes':'sum',
           'precisionF':'first',
           'numpictF':'first',
           'numlinksF':'first',
           'lengthF':'first'}

dfW = data.groupby(['PostId',data['periods']//7]).agg(aggdict) # aggregate every 7 days from creation, left-closed bins
dfW.rename(columns={'periods':'periods_orig_left'}, inplace=True)
dfW = dfW.reset_index()

# 1)  upVotes
dfW.loc[:,'up_lag'] = dfW.groupby('PostId')['numUpvotes'].transform(lambda x: x.shift(1))

meanXt = dfW.groupby('periods')['numUpvotes'].mean()

# model 1: exponential (pendulum)
def expo(t, A, tau): # exponential decay function; A = aplitude, tau = decay time
	return A * np.exp(-t/tau) 

par_upW, cov_upW = curve_fit(expo, dfW['periods'].values, dfW['numUpvotes'].values) # fit all data together with scipy curve_fit
pd.to_pickle(par_upW,directory + 'decay_params_up_week.pkl') # par[0]==estiamte of A, par[1]=estiamate of tau
'''
points_t = par[0] * np.exp(-t/par[1]) with the first period being t==0
'''

# model 2: AR(1) process
results = smf.ols('numUpvotes_sum ~ up_lag -1', data=dfW).fit()
print(results.summary())
beta_upW = results.params[0]
pd.to_pickle(beta_upW,directory + 'decay_params_ar1_up_week.pkl')

# model r: potential
def pot(t, A, k):
    return A*(t+1)**k
par_pot_upW, cov_pot_upW = curve_fit(pot, dfW['periods'].values, dfW['numUpvotes'].values)
pd.to_pickle(par_pot_upW,directory + 'decay_params_pot_up_week.pkl')

# plot two models for 10 first periods

def ar1(t, A, beta):
    return A * (beta**(t))

num_periods_toplot = 5

means = meanXt[:num_periods_toplot+1].values
t_means = np.arange(0,num_periods_toplot+1,step=1)
t_function = np.linspace(0, num_periods_toplot+1, 60)

plt.figure()
#plt.plot(t, x, 'b.', label='data')
plt.plot(t_means, means, 'ro', label='Data means')
plt.plot(t_function, expo(t_function, *par_upW), color='black',linestyle='solid', label=r'Exponential: up-votes = $\hat{A}e^{-\frac{t}{\hat{\tau}}} $')
plt.plot(t_function, ar1(t_function, means[0], beta_upW),color='black',linestyle='dashed', label=r'AR(1):  up-votes = $\hat{A} \hat{\gamma}^t $')
plt.plot(t_function, pot(t_function, *par_pot_upW),linestyle='dotted',color='black', label=r'Power: up-votes = $\hat{A}(t+1)^{\hat{\delta}} $')
plt.xlabel(r'weeks (t)') # x-label
plt.ylabel(r'Num. Upvotes') # y-label
plt.grid(True, which='both') # create background grid
plt.legend(loc=0, fontsize='xx-large') # put legend in authomatically chosen optimal location
plt.annotate("", xy=(-0.1, 0), xytext=(-0.1, means[0]),
             arrowprops=dict(arrowstyle="<->", connectionstyle="arc3"))
plt.text(-0.1, means[0]/2, r'$\hat{A}$',
         {'color': 'black', 'fontsize': 24, 'ha': 'center', 'va': 'center',
          'bbox': dict(boxstyle="round", fc="white", ec="black", pad=0.2)})
plt.tight_layout() # make a compact figure
# saved as Decay_upvotes_weeks.png

# 2) downVotes
dfW.loc[:,'down_lag'] = dfW.groupby('PostId')['numDownvotes'].transform(lambda x: x.shift(1))

meanXtd = dfW.groupby('periods')['numDownvotes'].mean()

# model 1: exponential (pendulum)
def expo(t, A, tau): # exponential decay function; A = aplitude, tau = decay time
	return A * np.exp(-t/tau) 

par_downW, cov_downW = curve_fit(expo, dfW['periods'].values, dfW['numDownvotes'].values) # fit all data together with scipy curve_fit
pd.to_pickle(par_downW,directory + 'decay_params_down_week.pkl') # par[0]==estiamte of A, par[1]=estiamate of tau
'''
points_t = par[0] * np.exp(-t/par[1]) with the first period being t==0
'''

# model 2: AR(1) process
results = smf.ols('numDownvotes_sum ~ down_lag -1', data=dfW).fit()
print(results.summary())
beta_downW = results.params[0]
pd.to_pickle(beta_downW,directory + 'decay_params_ar1_down_week.pkl')

# model r: potential
def pot(t, A, k):
    return A*(t+1)**k
par_pot_downW, cov_pot_downW = curve_fit(pot, dfW['periods'].values, dfW['numDownvotes'].values)
pd.to_pickle(par_pot_downW,directory + 'decay_params_pot_down_week.pkl')

# plot two models for 10 first periods

def ar1(t, A, beta):
    return A * (beta**(t))

num_periods_toplot = 5 

means = meanXtd[:num_periods_toplot+1].values
t_means = np.arange(0,num_periods_toplot+1,step=1)
t_function = np.linspace(0, num_periods_toplot+1, 60)

plt.figure()
#plt.plot(t, x, 'b.', label='data')
plt.plot(t_means, means, 'ro', label='Data means')
plt.plot(t_function, expo(t_function, *par_downW), color='black',linestyle='solid', label=r'Exponential: down-votes = $\hat{A}e^{-\frac{t}{\hat{\tau}}} $')
plt.plot(t_function, ar1(t_function, means[0], beta_downW),color='black',linestyle='dashed', label=r'AR(1): down-votes = $\hat{A} \hat{\gamma}^t $')
plt.plot(t_function, pot(t_function, *par_pot_downW),linestyle='dotted',color='black', label=r'Power: down-votes = $\hat{A}(t+1)^{\hat{\delta}} $')
plt.xlabel(r'weeks (t)') # x-label
plt.ylabel(r'Num. Downvotes') # y-label
plt.grid(True, which='both') # create background grid
plt.legend(loc=0, fontsize='xx-large') # put legend in authomatically chosen optimal location
plt.annotate("", xy=(-0.1, 0), xytext=(-0.1, means[0]),
             arrowprops=dict(arrowstyle="<->", connectionstyle="arc3"))
plt.text(-0.1, means[0]/2, r'$\hat{A}$',
         {'color': 'black', 'fontsize': 24, 'ha': 'center', 'va': 'center',
          'bbox': dict(boxstyle="round", fc="white", ec="black", pad=0.2)})
plt.tight_layout() # make a compact figure

# saved as Decay_downvotes_week.png




## ARE PARAMETERS DIFFERENT FOR DIFFERENT LEVELS OF EFFORT?
# create quality variable
dfW['points'] = (10*dfW['numUpvotes']) - (2*dfW['numDownvotes'])
dfW['precisionF2'] = dfW['precisionF']**2
dfW['lengthF2'] = dfW['lengthF']**2
dfW['numpictF2'] = dfW['numpictF']**2
dfW['numlinksF2'] = dfW['numlinksF']**2
modfit = smf.ols('points ~ precisionF + precisionF2 + lengthF + lengthF2 + numpictF + numpictF2 + numlinksF + numlinksF2',
        data=dfW).fit()
dfW['quality'] = modfit.predict(dfW[['precisionF','precisionF2','lengthF','lengthF2','numpictF','numpictF2','numlinksF','numlinksF2']])

subsampleA = dfW.loc[dfW['quality']<dfW['quality'].quantile(0.33),] # take subsample of up-votes within a selected quantile
subsampleB = dfW.loc[(dfW['quality']>=dfW['quality'].quantile(0.33)) & (dfW['quality']<dfW['quality'].quantile(0.66))  ,]
subsampleC = dfW.loc[(dfW['quality']>=dfW['quality'].quantile(0.66)),]

meanXtA = subsampleA.groupby('periods')['numUpvotes'].mean()
meanXtB = subsampleB.groupby('periods')['numUpvotes'].mean()
meanXtC = subsampleC.groupby('periods')['numUpvotes'].mean()

parAW, covAW = curve_fit(expo, subsampleA['periods'].values, subsampleA['numUpvotes'].values) # fit data A
parBW, covBW = curve_fit(expo, subsampleB['periods'].values, subsampleB['numUpvotes'].values) # fit data B
parCW, covCW = curve_fit(expo, subsampleC['periods'].values, subsampleC['numUpvotes'].values) # fit data C

meansA = meanXtA[:num_periods_toplot+1].values
meansB = meanXtB[:num_periods_toplot+1].values
meansC = meanXtC[:num_periods_toplot+1].values

plt.figure()
#plt.plot(t, x, 'b.', label='data')
plt.plot(t_means, meansA, 'r+', label='mean low quality')
plt.plot(t_means, meansB, 'rx', label='mean medium quality')
plt.plot(t_means, meansC, 'ro', label='mean high quality')
plt.plot(t_function, expo(t_function, *parAW), label='low quality')
plt.plot(t_function, expo(t_function, *parBW), label='medium quality')
plt.plot(t_function, expo(t_function, *parCW), label='high quality')
plt.xlabel(r'weeks') # x-label
plt.ylabel(r'Num Upvotes') # y-label
plt.grid(True, which='both') # create background grid
plt.legend(loc=0) # put legend in authomatically chosen optimal location
plt.tight_layout() # make a compact figure

# saved as Decay_upvotes_byquality_week.png

parDW, covDW = curve_fit(expo, subsampleA['periods'].values, subsampleA['numDownvotes'].values) # fit data A
parEW, covEW = curve_fit(expo, subsampleB['periods'].values, subsampleB['numDownvotes'].values) # fit data B
parFW, covFW = curve_fit(expo, subsampleC['periods'].values, subsampleC['numDownvotes'].values) # fit data C

meanXtA = subsampleA.groupby('periods')['numDownvotes'].mean()
meanXtB = subsampleB.groupby('periods')['numDownvotes'].mean()
meanXtC = subsampleC.groupby('periods')['numDownvotes'].mean()

meansA = meanXtA[:num_periods_toplot+1].values
meansB = meanXtB[:num_periods_toplot+1].values
meansC = meanXtC[:num_periods_toplot+1].values

plt.figure()
#plt.plot(t, x, 'b.', label='data')
plt.plot(t_means, meansA, 'r+', label='mean low quality')
plt.plot(t_means, meansB, 'rx', label='mean medium quality')
plt.plot(t_means, meansC, 'ro', label='mean high quality')
plt.plot(t_function, expo(t_function, *parDW), label='low quality')
plt.plot(t_function, expo(t_function, *parEW), label='medium quality')
plt.plot(t_function, expo(t_function, *parFW), label='high quality')
plt.xlabel(r'days') # x-label
plt.ylabel(r'Num Downvotes') # y-label
plt.grid(True, which='both') # create background grid
plt.legend(loc=0) # put legend in authomatically chosen optimal location
plt.tight_layout() # make a compact figure

# saved as Decay_downvotes_byquality_week.png
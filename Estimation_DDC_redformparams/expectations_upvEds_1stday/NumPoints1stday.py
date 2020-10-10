# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 12:23:05 2019

@author: jacopo
"""

'''
this code estimate average arrival of points on the first day of creation,
given different quality levels.

'''

import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# qa_name = 'apple/'
qa_name = 'ell/'

# server
directory_data = 'S:\\users\\jacopo\\Documents\\'
# local 
directory_data = '/home/jacopo/OneDrive/Dati_Jac_locali/stack/' + qa_name 

#colstouse = ['PostId','periods','OwnerUserId','precision','nostop','length','numedits_totalOthers','numUpvotes']
df = pd.read_csv(directory_data + 'postHistV2_1period.csv', dtype={'OwnerUserId':'str'}, parse_dates=['day'])

''' NOT ADDED , TOO MANY
# load topics (tags)
# server
answerTags = pd.read_csv(r'\\tsclient\jacopo\OneDrive\Dati_Jac_locali\stack\apple\Answer2Tags.csv')
# local
answerTags = pd.read_csv(directory_data + 'Answer2Tags.csv')
answerTags = answerTags.rename(columns={'Id':'PostId'})
df = pd.merge(df, answerTags, on='PostId', validate='1:1', how='inner')
'''
# load user experience
userexp = pd.read_csv(directory_data + 'UserExperience2Answer.csv', dtype={'OwnerUserId':'str'})
userexp = userexp.rename(columns={'Id':'PostId'})
# when post is published before private beta --> negative Seniority_days
# set to 0 in this case
userexp.loc[userexp['Seniority_days']<0,'Seniority_days'] = 0
df = pd.merge(df, userexp, on='PostId', validate='1:1', how='inner') # around 2000 obs dropped 

# add year
df['year'] = df['day'].apply(lambda x: x.year)

# edits received and implemented
df['numedits_totalothers_accepted'] = df['numedits_sAOthers']  + df['numedits_dOthers']

# quality measure
df['points'] = (10*df['numUpvotes']) - (2*df['numDownvotes'])
df['precisionF2'] = df['precisionF']**2
df['lengthF2'] = df['lengthF']**2
df['numpictF2'] = df['numpictF']**2
df['numlinksF2'] = df['numlinksF']**2
modfit = smf.ols('points ~ precisionF + precisionF2 + lengthF + lengthF2 + numpictF + numpictF2 + numlinksF + numlinksF2',
        data=df).fit()
pd.to_pickle(modfit, directory_data + 'qualityfit.pkl')

df['quality'] = modfit.predict(df[['precisionF','precisionF2','lengthF','lengthF2','numpictF','numpictF2','numlinksF','numlinksF2']])


'''
agent learns from his actions or from community outputs?
if 1) agent learns from his action, then i should put expectations over arrival of edits
if 2) agent learns from community, use instead the realized number of edits received.
--> i do 2. (Note! when users form expectation of point arrival, they need first to make expectation on the number of 
edits that they will receive. BUT the model parameters are based on inference that they do on the whole community
activity, so the estimation of the model uses the truly realized edits)
for number 1, create the var "expected_edits" and substitute "numedits_totalOthers" with "expected_edits".
# load model for expectations over arrival of edits (built in script: NumEdits1stday.py)
expect_edits = pd.read_pickle(directory_data + '\PoissonReg_EdvsQ_noTopics.pkl')
df['expected_edits'] = expect_edits.predict(df)
'''



##############################################################################
#       models
##############################################################################
# Note: final specification does not include years: years FE have high coefs but not significant, so risk they distort too much prediction 
results4B = smf.glm("numUpvotes ~ quality + AnswerNum + Seniority_days + numedits_totalothers_accepted", data = df, family=sm.families.Poisson()).fit()
print(results4B.summary())
pd.to_pickle(results4B, directory_data + 'PoissonReg_UpvsQ_noTopics.pkl')

#pd.to_pickle(results4B, r'\\tsclient\jacopo\OneDrive\Dati_Jac_locali\stack\apple\PoissonReg_UpvsQ_noTopics.pkl')

'''
                 Generalized Linear Model Regression Results                  
==============================================================================
Dep. Variable:             numUpvotes   No. Observations:               118552
Model:                            GLM   Df Residuals:                   118547
Model Family:                 Poisson   Df Model:                            4
Link Function:                    log   Scale:                          1.0000
Method:                          IRLS   Log-Likelihood:            -2.4050e+05
Date:                sab, 13 giu 2020   Deviance:                   2.9683e+05
Time:                        00:26:23   Pearson chi2:                 4.93e+05
No. Iterations:                     6                                         
Covariance Type:            nonrobust                                         
=================================================================================================
                                    coef    std err          z      P>|z|      [0.025      0.975]
-------------------------------------------------------------------------------------------------
Intercept                        -0.4687      0.012    -39.587      0.000      -0.492      -0.446
quality                           0.0513      0.001     67.065      0.000       0.050       0.053
AnswerNum                      4.652e-05   3.95e-06     11.793      0.000    3.88e-05    5.43e-05
Seniority_days                    0.0001   4.64e-06     24.036      0.000       0.000       0.000
numedits_totalothers_accepted     0.4849      0.008     58.789      0.000       0.469       0.501
=================================================================================================
'''
results4C = smf.glm("numDownvotes ~ quality + AnswerNum + Seniority_days + numedits_totalothers_accepted", data = df, family=sm.families.Poisson()).fit()
print(results4C.summary())
pd.to_pickle(results4C, directory_data + 'PoissonReg_DownvsQ_noTopics.pkl')

'''
                 Generalized Linear Model Regression Results                  
==============================================================================
Dep. Variable:           numDownvotes   No. Observations:               118552
Model:                            GLM   Df Residuals:                   118547
Model Family:                 Poisson   Df Model:                            4
Link Function:                    log   Scale:                          1.0000
Method:                          IRLS   Log-Likelihood:                -38573.
Date:                sab, 13 giu 2020   Deviance:                       57874.
Time:                        00:42:27   Pearson chi2:                 1.62e+05
No. Iterations:                     7                                         
Covariance Type:            nonrobust                                         
=================================================================================================
                                    coef    std err          z      P>|z|      [0.025      0.975]
-------------------------------------------------------------------------------------------------
Intercept                        -1.5280      0.054    -28.435      0.000      -1.633      -1.423
quality                          -0.0488      0.004    -13.271      0.000      -0.056      -0.042
AnswerNum                        -0.0002   2.17e-05    -10.845      0.000      -0.000      -0.000
Seniority_days                   -0.0002   2.21e-05    -10.793      0.000      -0.000      -0.000
numedits_totalothers_accepted     0.6654      0.027     24.282      0.000       0.612       0.719
=================================================================================================
'''

results5B1 = smf.ols("numUpvotes ~ quality + AnswerNum + Seniority_days + C(year) + numedits_totalothers_accepted", data = df).fit()
print(results5B1.summary())
'''
                            OLS Regression Results                            
==============================================================================
Dep. Variable:             numUpvotes   R-squared:                       0.047
Model:                            OLS   Adj. R-squared:                  0.047
Method:                 Least Squares   F-statistic:                     458.5
Date:                ven, 01 mag 2020   Prob (F-statistic):               0.00
Time:                        19:53:56   Log-Likelihood:            -1.6467e+05
No. Observations:              121523   AIC:                         3.294e+05
Df Residuals:                  121509   BIC:                         3.295e+05
Df Model:                          13                                         
Covariance Type:            nonrobust                                         
=================================================================================================
                                    coef    std err          t      P>|t|      [0.025      0.975]
-------------------------------------------------------------------------------------------------
Intercept                        -0.0362      0.383     -0.094      0.925      -0.787       0.715
C(year)[T.2009]                   0.3488      0.401      0.870      0.385      -0.437       1.135
C(year)[T.2010]                   0.2612      0.383      0.681      0.496      -0.490       1.013
C(year)[T.2011]                   0.1405      0.383      0.367      0.714      -0.610       0.891
C(year)[T.2012]                   0.0472      0.383      0.123      0.902      -0.704       0.798
C(year)[T.2013]                  -0.0214      0.383     -0.056      0.956      -0.772       0.729
C(year)[T.2014]                  -0.0211      0.383     -0.055      0.956      -0.772       0.730
C(year)[T.2015]                  -0.0576      0.383     -0.150      0.880      -0.808       0.693
C(year)[T.2016]                  -0.0673      0.383     -0.176      0.861      -0.818       0.684
C(year)[T.2017]                   0.0386      0.383      0.101      0.920      -0.712       0.790
quality                           0.0878      0.004     22.774      0.000       0.080       0.095
AnswerNum                      2.013e-05   5.23e-06      3.853      0.000    9.89e-06    3.04e-05
Seniority_days                 8.887e-05   6.79e-06     13.096      0.000    7.56e-05       0.000
numedits_totalothers_accepted     1.4488      0.022     66.740      0.000       1.406       1.491
==============================================================================
Omnibus:                   487151.783   Durbin-Watson:                   1.918
Prob(Omnibus):                  0.000   Jarque-Bera (JB):    2519921339398.458
Skew:                         102.430   Prob(JB):                         0.00
Kurtosis:                   22310.534   Cond. No.                     3.42e+05
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 3.42e+05. This might indicate that there are
strong multicollinearity or other numerical problems.
'''

##### old stuff


# median value of quality for post at creation day
medPrec = df1period['precision'].quantile(0.5)

df1period.loc[df1period['precision']>=medPrec,'Quality'] = 'High'
df1period.loc[:,'Quality'].fillna('Low',inplace=True)

mean = df1period['numUpvotes'].mean()
std = df1period['numUpvotes'].std()
df1period.loc[:,'numUpvotes_std'] = df1period['numUpvotes'].apply(lambda x: (x -mean )/std)
meanP = df1period['precision'].mean()
stdP = df1period['precision'].std()
df1period.loc[:,'p_std'] = df1period['precision'].apply(lambda x: (x -meanP )/stdP)


results = smf.glm("numUpvotes ~ C(Quality) + C(TAG)", data = df1period, family=sm.families.Poisson()).fit()
with open('results.csv', 'w') as f:
    f.write(results.summary().as_csv())

resultsbis = smf.glm("numUpvotes ~ C(Quality) + C(TAG) + AnswerNum + Seniority_days", data = df1period, family=sm.families.Poisson()).fit()
with open('results2.csv', 'w') as f:
    f.write(resultsbis.summary().as_csv())
newobs = pd.DataFrame({'Quality':['Low','High'], 'TAG':['emoji','emoji'], 'AnswerNum':[1,1],'Seniority_days':[1,1]})
resultsbis.predict(newobs)

resultster = smf.ols("numUpvotes ~ C(Quality) + C(TAG) + AnswerNum + Seniority_days", data = df1period).fit()
with open('results3.csv', 'w') as f:
    f.write(resultster.summary().as_csv())

results2 = smf.glm("numUpvotes ~ C(Quality)", data = df1period, family=sm.families.Poisson()).fit()
print(results2.summary())

results3 = smf.ols("numUpvotes ~ C(Quality)", data = df1period).fit()
print(results3.summary())

##############################################################################
#       quality = LOW, HIGH (precision below/above median)
##############################################################################

results4 = smf.glm("numUpvotes ~ C(Quality) + AnswerNum + Seniority_days + numedits_totalOthers", data = df1period, family=sm.families.Poisson()).fit()
print(results4.summary())
'''
                 Generalized Linear Model Regression Results                  
==============================================================================
Dep. Variable:             numUpvotes   No. Observations:                90105
Model:                            GLM   Df Residuals:                    90100
Model Family:                 Poisson   Df Model:                            4
Link Function:                    log   Scale:                          1.0000
Method:                          IRLS   Log-Likelihood:            -1.2080e+05
Date:                Sun, 05 May 2019   Deviance:                   1.3564e+05
Time:                        15:10:47   Pearson chi2:                 1.71e+05
No. Iterations:                     5   Covariance Type:             nonrobust
========================================================================================
                           coef    std err          z      P>|z|      [0.025      0.975]
----------------------------------------------------------------------------------------
Intercept               -0.2277      0.006    -37.892      0.000      -0.239      -0.216
C(Quality)[T.Low]       -0.1112      0.007    -15.390      0.000      -0.125      -0.097
AnswerNum               9.9e-05   5.99e-06     16.524      0.000    8.73e-05       0.000
Seniority_days           0.0002   8.23e-06     20.071      0.000       0.000       0.000
numedits_totalOthers     0.3862      0.008     45.698      0.000       0.370       0.403
========================================================================================
'''

newobs = pd.DataFrame({'Quality':['Low'],'AnswerNum':[1],'Seniority_days':[1]})
results4.predict(newobs) # equiv to: np.exp(-0.1702 + -0.1209 * 1 + 0.00009493 * 1 + 0.0001*1)


results5 = smf.ols("numUpvotes ~ C(Quality) + AnswerNum + Seniority_days + numedits_totalOthers", data = df1period).fit()
print(results5.summary())
'''
                            OLS Regression Results                            
==============================================================================
Dep. Variable:             numUpvotes   R-squared:                       0.019
Model:                            OLS   Adj. R-squared:                  0.019
Method:                 Least Squares   F-statistic:                     446.6
Date:                Sun, 05 May 2019   Prob (F-statistic):               0.00
Time:                        15:26:39   Log-Likelihood:            -1.5129e+05
No. Observations:               90105   AIC:                         3.026e+05
Df Residuals:                   90100   BIC:                         3.026e+05
Df Model:                           4                                         
Covariance Type:            nonrobust                                         
========================================================================================
                           coef    std err          t      P>|t|      [0.025      0.975]
----------------------------------------------------------------------------------------
Intercept                0.7906      0.007    109.086      0.000       0.776       0.805
C(Quality)[T.Low]       -0.0948      0.009    -10.945      0.000      -0.112      -0.078
AnswerNum                0.0001   8.54e-06     12.854      0.000     9.3e-05       0.000
Seniority_days           0.0002   1.05e-05     14.366      0.000       0.000       0.000
numedits_totalOthers     0.4371      0.013     32.402      0.000       0.411       0.464
==============================================================================
Omnibus:                    82447.080   Durbin-Watson:                   1.994
Prob(Omnibus):                  0.000   Jarque-Bera (JB):          6660067.129
Skew:                           4.145   Prob(JB):                         0.00
Kurtosis:                      44.294   Cond. No.                     2.38e+03
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 2.38e+03. This might indicate that there are
strong multicollinearity or other numerical problems.
'''


##############################  OTHER STUFF

fig = sns.pointplot(x='Quality', y='numUpvotes', data=df1period , linestyles='--')
plt.ylabel('avg number of up-votes')
plt.title('Average number of up-votes at creation date')
fig.get_figure().savefig('figures\\QualityReturns_1stday_2levels.png', dpi=500)

bins = [0,df1period['precision'].quantile(0.25),df1period['precision'].quantile(0.5),df1period['precision'].quantile(0.75)]
df1period.loc[:,'precision_bins'] = pd.Series(np.digitize(df1period['precision'], bins), index=df1period.index)
d = {1:'[0,Q(.25)[', 2:'[Q(.25),Q(.5)[', 3:'[Q(.5),Q(.75)[', 4:'>=Q(.75)'}
df1period.loc[:,'precision_bins'] = df1period['precision_bins'].apply(lambda x: d[x])
fig2 = sns.pointplot(x='precision_bins', y='numUpvotes', data=df1period, linestyles='--')
plt.xlabel('Precision')
plt.ylabel('avg number of up-votes')
plt.title('Average number of up-votes at creation date')
fig2.get_figure().savefig('figures\\QualityReturns_1stday_4levels.png', dpi=500)


df1period.loc[df1period['nostop']>=df1period['nostop'].median(),'QualityNS'] = 'High'
df1period.loc[:,'QualityNS'].fillna('Low',inplace=True)
sns.pointplot(x='QualityNS', y='numUpvotes', data=df1period )

bins = [0,df1period['nostop'].quantile(0.25),df1period['nostop'].quantile(0.5),df1period['nostop'].quantile(0.75)]
df1period.loc[:,'nostop_bins'] = pd.Series(np.digitize(df1period['nostop'], bins), index=df1period.index)
sns.pointplot(x='nostop_bins', y='numUpvotes', data=df1period )

sns.distplot(df1period['nostop'])
plt.yscale('log')

bins = [0,df1period['length'].quantile(0.25),df1period['length'].quantile(0.5),df1period['length'].quantile(0.75)]
df1period.loc[:,'length_bins'] = pd.Series(np.digitize(df1period['length'], bins), index=df1period.index)
sns.pointplot(x='length_bins', y='numUpvotes', data=df1period )


results6 = smf.ols("numUpvotes ~ precision_std + AnswerNum + Seniority_days", data = df1period).fit()
print(results6.summary())
'''
                            OLS Regression Results                            
==============================================================================
Dep. Variable:             numUpvotes   R-squared:                       0.008
Model:                            OLS   Adj. R-squared:                  0.008
Method:                 Least Squares   F-statistic:                     240.9
Date:                Sun, 31 Mar 2019   Prob (F-statistic):          9.95e-156
Time:                        11:25:09   Log-Likelihood:            -1.5182e+05
No. Observations:               90105   AIC:                         3.036e+05
Df Residuals:                   90101   BIC:                         3.037e+05
Df Model:                           3                                         
Covariance Type:            nonrobust                                         
==================================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
----------------------------------------------------------------------------------
Intercept          0.7914      0.005    144.237      0.000       0.781       0.802
precision_std      0.0509      0.004     11.694      0.000       0.042       0.059
AnswerNum          0.0001   8.58e-06     12.164      0.000    8.76e-05       0.000
Seniority_days     0.0001   1.05e-05     12.283      0.000       0.000       0.000
==============================================================================
Omnibus:                    83142.153   Durbin-Watson:                   1.992
Prob(Omnibus):                  0.000   Jarque-Bera (JB):          6771936.177
Skew:                           4.202   Prob(JB):                         0.00
Kurtosis:                      44.631   Cond. No.                         950.
=============================================================================='''

results7 = smf.glm("numUpvotes ~ precision_std + AnswerNum + Seniority_days", data = df1period, family=sm.families.Poisson()).fit()
print(results7.summary())
'''
                 Generalized Linear Model Regression Results                  
==============================================================================
Dep. Variable:             numUpvotes   No. Observations:                90105
Model:                            GLM   Df Residuals:                    90101
Model Family:                 Poisson   Df Model:                            3
Link Function:                    log   Scale:                          1.0000
Method:                          IRLS   Log-Likelihood:            -1.2166e+05
Date:                Thu, 04 Apr 2019   Deviance:                   1.3736e+05
Time:                        16:27:29   Pearson chi2:                 1.77e+05
No. Iterations:                     5   Covariance Type:             nonrobust
==================================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
----------------------------------------------------------------------------------
Intercept         -0.2307      0.005    -49.866      0.000      -0.240      -0.222
precision_std      0.0586      0.004     16.662      0.000       0.052       0.066
AnswerNum        9.37e-05   6.01e-06     15.581      0.000    8.19e-05       0.000
Seniority_days     0.0001   8.25e-06     17.555      0.000       0.000       0.000
==================================================================================
'''
results8 = smf.glm("numUpvotes ~ nostop + AnswerNum + Seniority_days", data = df1period, family=sm.families.Poisson()).fit()
print(results8.summary())

results9 = smf.ols("numUpvotes ~ nostop + AnswerNum + Seniority_days", data = df1period).fit()
print(results9.summary())

results10 = smf.glm("numUpvotes ~ precision + AnswerNum + Seniority_days", data = df1period, family=sm.families.Poisson()).fit()
print(results10.summary())
newobs = pd.DataFrame({'precision':[0.65833],'AnswerNum':[0],'Seniority_days':[0]})
results10.predict(newobs)
pd.to_pickle(results10, r'\\tsclient\jacopo\OneDrive\Dati_Jac_locali\stack\apple\PoissonReg_UpvsQ_prec_noTopics.pkl')

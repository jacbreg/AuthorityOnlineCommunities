# -*- coding: utf-8 -*-
"""
Created on Sun May  5 16:12:51 2019

@author: jacopo
"""

import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# qa_name = 'apple/'
qa_name = 'ell/'
directory_data = '/home/jacopo/OneDrive/Dati_Jac_locali/stack/' + qa_name 

df = pd.read_csv(directory_data + 'postHistV2_1period.csv', dtype={'OwnerUserId':'str'}, parse_dates=['day'])


# load user experience
# server
userexp = pd.read_csv(r'\\tsclient\jacopo\OneDrive\Dati_Jac_locali\stack\apple\UserExperience2Answer.csv')
#local
userexp = pd.read_csv(directory_data + 'UserExperience2Answer.csv')
userexp = userexp.rename(columns={'Id':'PostId'})
# when post is published before private beta --> negative Seniority_days
# set to 0 in this case
userexp.loc[userexp['Seniority_days']<0,'Seniority_days'] = 0
df = pd.merge(df, userexp, on='PostId', validate='1:1', how='inner') # around 2000 obs dropped (had seniority<0?)

## add year
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
df['quality'] = modfit.predict(df[['precisionF','precisionF2','lengthF','lengthF2','numpictF','numpictF2','numlinksF','numlinksF2']])


##############################################################################
#       quality = precision
##############################################################################
# Note: final specification does not include years: years FE have high coefs but not significant, so risk they distort too much prediction 
results3 = smf.glm("numedits_totalothers_accepted ~ quality + AnswerNum + Seniority_days", data = df, family=sm.families.Poisson()).fit()
print(results3.summary())
pd.to_pickle(results3, directory_data + 'PoissonReg_EdvsQ_noTopics.pkl')

pd.to_pickle(results3, r'\\tsclient\jacopo\OneDrive\Dati_Jac_locali\stack\apple\PoissonReg_EdvsQ_noTopics.pkl')
'''
ELL
                       Generalized Linear Model Regression Results                       
=========================================================================================
Dep. Variable:     numedits_totalothers_accepted   No. Observations:               118552
Model:                                       GLM   Df Residuals:                   118548
Model Family:                            Poisson   Df Model:                            3
Link Function:                               log   Scale:                          1.0000
Method:                                     IRLS   Log-Likelihood:                -21008.
Date:                           sab, 13 giu 2020   Deviance:                       32276.
Time:                                   00:15:47   Pearson chi2:                 1.33e+05
No. Iterations:                                7                                         
Covariance Type:                       nonrobust                                         
==================================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
----------------------------------------------------------------------------------
Intercept         -2.7876      0.087    -32.119      0.000      -2.958      -2.617
quality           -0.0018      0.006     -0.302      0.763      -0.013       0.010
AnswerNum         -0.0004   3.99e-05     -9.585      0.000      -0.000      -0.000
Seniority_days    -0.0006   3.85e-05    -15.847      0.000      -0.001      -0.001
==================================================================================
'''

results4 = smf.ols("numedits_totalothers_accepted ~ quality + AnswerNum + Seniority_days", data = df).fit()
print(results4.summary())
'''
                                  OLS Regression Results                                 
=========================================================================================
Dep. Variable:     numedits_totalothers_accepted   R-squared:                       0.005
Model:                                       OLS   Adj. R-squared:                  0.005
Method:                            Least Squares   F-statistic:                     210.2
Date:                           sab, 13 giu 2020   Prob (F-statistic):          5.15e-136
Time:                                   00:21:37   Log-Likelihood:                 12668.
No. Observations:                         118552   AIC:                        -2.533e+04
Df Residuals:                             118548   BIC:                        -2.529e+04
Df Model:                                      3                                         
Covariance Type:                       nonrobust                                         
==================================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
----------------------------------------------------------------------------------
Intercept          0.0598      0.004     15.621      0.000       0.052       0.067
quality           -0.0002      0.000     -0.872      0.383      -0.001       0.000
AnswerNum      -8.706e-06   1.11e-06     -7.854      0.000   -1.09e-05   -6.53e-06
Seniority_days -2.075e-05   1.29e-06    -16.065      0.000   -2.33e-05   -1.82e-05
==============================================================================
Omnibus:                   123810.075   Durbin-Watson:                   1.965
Prob(Omnibus):                  0.000   Jarque-Bera (JB):          6286136.652
Skew:                           5.452   Prob(JB):                         0.00
Kurtosis:                      36.966   Cond. No.                     5.97e+03
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 5.97e+03. This might indicate that there are
strong multicollinearity or other numerical problems.
'''


# old

##############################################################################
#       quality = precision std
##############################################################################

meanP = df1period['precision'].mean()
stdP = df1period['precision'].std()
df1period.loc[:,'p_std'] = df1period['precision'].apply(lambda x: (x -meanP )/stdP)

results5 = smf.glm("numedits_totalOthers ~ p_std + AnswerNum + Seniority_days", data = df1period, family=sm.families.Poisson()).fit()
print(results5.summary())

'''
                  Generalized Linear Model Regression Results                   
================================================================================
Dep. Variable:     numedits_totalOthers   No. Observations:                90105
Model:                              GLM   Df Residuals:                    90101
Model Family:                   Poisson   Df Model:                            3
Link Function:                      log   Scale:                          1.0000
Method:                            IRLS   Log-Likelihood:                -27777.
Date:                  Mon, 06 May 2019   Deviance:                       40456.
Time:                          17:28:20   Pearson chi2:                 1.06e+05
No. Iterations:                       6   Covariance Type:             nonrobust
==================================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
----------------------------------------------------------------------------------
Intercept         -2.1678      0.013   -162.119      0.000      -2.194      -2.142
p_std              0.1471      0.010     14.387      0.000       0.127       0.167
AnswerNum         -0.0003   4.03e-05     -7.457      0.000      -0.000      -0.000
Seniority_days    -0.0007   3.58e-05    -20.075      0.000      -0.001      -0.001
==================================================================================
'''

results6 = smf.ols("numedits_totalOthers ~ p_std + AnswerNum + Seniority_days", data = df1period).fit()
print(results6.summary())
'''
                             OLS Regression Results                             
================================================================================
Dep. Variable:     numedits_totalOthers   R-squared:                       0.009
Model:                              OLS   Adj. R-squared:                  0.009
Method:                   Least Squares   F-statistic:                     265.7
Date:                  Mon, 06 May 2019   Prob (F-statistic):          1.00e-171
Time:                          17:30:13   Log-Likelihood:                -25244.
No. Observations:                 90105   AIC:                         5.050e+04
Df Residuals:                     90101   BIC:                         5.053e+04
Df Model:                             3                                         
Covariance Type:              nonrobust                                         
==================================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
----------------------------------------------------------------------------------
Intercept          0.1108      0.001     82.292      0.000       0.108       0.113
p_std              0.0142      0.001     13.295      0.000       0.012       0.016
AnswerNum      -9.241e-06   2.11e-06     -4.388      0.000   -1.34e-05   -5.11e-06
Seniority_days -5.161e-05   2.59e-06    -19.937      0.000   -5.67e-05   -4.65e-05
==============================================================================
Omnibus:                    76768.232   Durbin-Watson:                   2.000
Prob(Omnibus):                  0.000   Jarque-Bera (JB):          2190748.747
Skew:                           4.101   Prob(JB):                         0.00
Kurtosis:                      25.721   Cond. No.                         950.
==============================================================================
'''
##############################################################################
#       quality = LOW, HIGH (precision below/above median)
##############################################################################

# median value of quality for post at creation day
medPrec = df1period['precision'].quantile(0.5)

df1period.loc[df1period['precision']>=medPrec,'Quality'] = 'High'
df1period.loc[:,'Quality'].fillna('Low',inplace=True)

results = smf.glm("numedits_totalOthers ~ C(Quality) + AnswerNum + Seniority_days", data = df1period, family=sm.families.Poisson()).fit()
print(results.summary())
'''
                  Generalized Linear Model Regression Results                   
================================================================================
Dep. Variable:     numedits_totalOthers   No. Observations:                90105
Model:                              GLM   Df Residuals:                    90101
Model Family:                   Poisson   Df Model:                            3
Link Function:                      log   Scale:                          1.0000
Method:                            IRLS   Log-Likelihood:                -27822.
Date:                  Sun, 05 May 2019   Deviance:                       40545.
Time:                          16:16:35   Pearson chi2:                 1.07e+05
No. Iterations:                       6   Covariance Type:             nonrobust
=====================================================================================
                        coef    std err          z      P>|z|      [0.025      0.975]
-------------------------------------------------------------------------------------
Intercept            -2.0488      0.017   -121.164      0.000      -2.082      -2.016
C(Quality)[T.Low]    -0.2312      0.022    -10.357      0.000      -0.275      -0.187
AnswerNum            -0.0003   4.01e-05     -7.385      0.000      -0.000      -0.000
Seniority_days       -0.0007   3.58e-05    -20.040      0.000      -0.001      -0.001
=====================================================================================
'''
results2 = smf.ols("numedits_totalOthers ~ C(Quality) + AnswerNum + Seniority_days", data = df1period).fit()
print(results2.summary())
'''
                             OLS Regression Results                             
================================================================================
Dep. Variable:     numedits_totalOthers   R-squared:                       0.008
Model:                              OLS   Adj. R-squared:                  0.008
Method:                   Least Squares   F-statistic:                     236.5
Date:                  Sun, 05 May 2019   Prob (F-statistic):          6.95e-153
Time:                          16:18:27   Log-Likelihood:                -25287.
No. Observations:                 90105   AIC:                         5.058e+04
Df Residuals:                     90101   BIC:                         5.062e+04
Df Model:                             3                                         
Covariance Type:              nonrobust                                         
=====================================================================================
                        coef    std err          t      P>|t|      [0.025      0.975]
-------------------------------------------------------------------------------------
Intercept             0.1208      0.002     69.270      0.000       0.117       0.124
C(Quality)[T.Low]    -0.0203      0.002     -9.474      0.000      -0.024      -0.016
AnswerNum         -9.203e-06   2.11e-06     -4.366      0.000   -1.33e-05   -5.07e-06
Seniority_days    -5.124e-05   2.59e-06    -19.779      0.000   -5.63e-05   -4.62e-05
==============================================================================
Omnibus:                    76882.039   Durbin-Watson:                   2.000
Prob(Omnibus):                  0.000   Jarque-Bera (JB):          2208204.883
Skew:                           4.107   Prob(JB):                         0.00
Kurtosis:                      25.819   Cond. No.                     1.77e+03
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 1.77e+03. This might indicate that there are
strong multicollinearity or other numerical problems.
'''

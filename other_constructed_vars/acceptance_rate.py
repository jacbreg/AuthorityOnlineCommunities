#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 16:37:02 2020

@author: jacopo
"""

'''
accpetance rate of suggested edits
'''

import pandas as pd

qa_name = 'ell/'

##########################################################################

directory = '/home/jacopo/OneDrive/Dati_Jac_locali/stack/' + qa_name

data = pd.read_csv(directory + 'postHistV2.csv', dtype={'OwnerUserId':str}, parse_dates=['day'])

data['numedits_sA'].sum() / (data['numedits_sA'] + data['numedits_sR']).sum() # 0.7377 --> 74%


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 14:48:02 2020

@author: jacopo
"""

'''
timeline of a period in the DDC model
'''

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# timeline di un period nel DDC model
plt.figure()
plt.xlim([0,1])
plt.ylim([0.2,0.8])
plt.arrow(0,0.5,1,0, length_includes_head=True, width=0.001, head_width=0.02,color='k')
plt.vlines([0.05,0.2,0.4,0.6,0.8,0.95], 
           [0.48,0.4,0.501,0.4,0.501,0.48],
           [0.52,0.499,0.6,0.499,0.6,0.52],
           colors=['k','lightgray','lightgray','lightgray','lightgray','k'])
plt.text(0.045, 0.44, 't')
plt.text(0.92, 0.44, 't+1')
plt.text(0.2,0.38, 'User\nobserves\nthe states', ha='center', va='top')
plt.text(0.4,0.62, 'User\nimplements choice', ha='center', va='bottom')
plt.text(0.6,0.38, 'Flow Payoff\nrealises', ha='center', va='top')
plt.text(0.8,0.62, 'New state values\nrealise', ha='center', va='bottom')
plt.axis('off')
# saved as timeline.png

# timeline of platform history
graduation = pd.Timestamp(2016,2,25).to_datetime64() #change of rep points https://ell.meta.stackexchange.com/questions/2945/new-site-design
actualgraduation = pd.Timestamp(2015,9,10).to_datetime64() # formally real graduation, but points don't change yet until design
public_beta_start = pd.Timestamp(2013,1,30).to_datetime64()
private_beta_start = pd.Timestamp(2013,1,23).to_datetime64()
electionsrelated = pd.read_csv('/home/jacopo/OneDrive/Dati_Jac_locali/stack/ell/elections.csv', dtype={'user':str}, parse_dates=['election began','election ended'])
electionsrelated = electionsrelated[['election began','election ended']].drop_duplicates()
download_date = pd.Timestamp(2020,5,31)
dates = pd.date_range(start=private_beta_start, end=download_date, freq='D')

plt.figure()
plt.ylim([-1,1])
plt.plot(dates,np.zeros(len(dates)), color='k')
plt.plot([private_beta_start, public_beta_start, actualgraduation, graduation],np.zeros(4),'-o', color='k')
plt.vlines([private_beta_start, public_beta_start, actualgraduation, graduation], 0 ,[-0.3,0.3,-0.3,0.3])
plt.text(private_beta_start, -0.29, 'Private Beta\nstarts', ha='center', va='top')
plt.text(public_beta_start, 0.31, 'Public Beta\nstarts', ha='center', va='bottom')
plt.text(actualgraduation, -0.29, 'Graduation', ha='center', va='top')
plt.text(graduation, 0.31, 'New design\nand change in\nreputation thresholds', ha='center', va='bottom')
for i in range(len(electionsrelated)):
    if i==0:
        plt.axvspan(electionsrelated['election began'].iloc[i], electionsrelated['election ended'].iloc[i], facecolor='lightgray', label='elections')
    else:
        plt.axvspan(electionsrelated['election began'].iloc[i], electionsrelated['election ended'].iloc[i], facecolor='lightgray')
plt.legend()
ax = plt.gca()
ax.get_yaxis().set_visible(False)
for spine in ["left", "top", "right"]:
    ax.spines[spine].set_visible(False)
# saved as history_platform.png

# incentive effects
from matplotlib import rc, rcParams
rc('text', usetex=True)
rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
plt.figure()
plt.xlim([0,1])
plt.ylim([0.4,0.77])
plt.arrow(0,0.5,1,0, length_includes_head=True, width=0.001, head_width=0.02,color='k')
plt.arrow(0.5,0.7,0,-0.18, length_includes_head=True, width=0.001, head_width=0.02,color='k')
plt.vlines(0.5,0.49,0.51, color='k')
plt.text(0.5,0.72,'Performance Threshold',horizontalalignment='center')
plt.text(0,0.48,'0',verticalalignment='top')
plt.text(0.9,0.48,'Individual performance\nmeasure\n(likes, reputation points, etc.)\nas outcome of task 2',horizontalalignment='center',verticalalignment='top')
plt.text(0.5,0.48,r'$T$',horizontalalignment='center',verticalalignment='top')
plt.text(0.25,0.55,r'$\overbrace{\hspace{160pt}}$',horizontalalignment='center')
plt.text(0.75,0.55,r'$\overbrace{\hspace{160pt}}$',horizontalalignment='center')
plt.text(0.25, 0.6, r'Incentive effect (?)',horizontalalignment='center')
plt.text(0.75, 0.6, r'long term effect (?)',horizontalalignment='center')
plt.axis('off')
plt.title('Delegation of authority on task 1')
plt.tight_layout()
# saved as incentive_effect.png

plt.figure()
plt.xlim([0,1])
plt.ylim([0.4,0.77])
plt.arrow(0,0.5,1,0, length_includes_head=True, width=0.001, head_width=0.02,color='k')
plt.arrow(0.5,0.7,0,-0.18, length_includes_head=True, width=0.001, head_width=0.02,color='k')
plt.vlines(0.5,0.49,0.51, color='k')
plt.text(0.5,0.72,'Performance Threshold',horizontalalignment='center')
plt.text(0,0.48,'0',verticalalignment='top')
plt.text(0.9,0.48,'Individual performance\nmeasure\n(likes, reputation points, etc.)\nas outcome of task 2',horizontalalignment='center',verticalalignment='top')
plt.text(0.5,0.48,r'$T$',horizontalalignment='center',verticalalignment='top')
plt.axis('off')
plt.title('Delegation of authority on task 1')
plt.tight_layout()
# saved as incentive_effect0.png

plt.figure()
plt.xlim([0,1])
plt.ylim([0.4,0.77])
plt.arrow(0,0.5,1,0, length_includes_head=True, width=0.001, head_width=0.02,color='k')
plt.arrow(0.5,0.7,0,-0.18, length_includes_head=True, width=0.001, head_width=0.02,color='k')
plt.vlines(0.5,0.49,0.51, color='k')
plt.text(0.5,0.72,'Performance Threshold',horizontalalignment='center')
plt.text(0,0.48,'0',verticalalignment='top')
plt.text(0.9,0.48,'Individual performance\nmeasure\n(likes, reputation points, etc.)\nas outcome of task 2',horizontalalignment='center',verticalalignment='top')
plt.text(0.5,0.48,r'$T$',horizontalalignment='center',verticalalignment='top')
plt.text(0.25,0.55,r'$\overbrace{\hspace{160pt}}$',horizontalalignment='center')
plt.text(0.25, 0.6, r'Incentive effect (?)',horizontalalignment='center')
plt.axis('off')
plt.title('Delegation of authority on task 1')
plt.tight_layout()
# saved as incentive_effect1.png

plt.figure()
plt.xlim([0,1])
plt.ylim([0.4,0.77])
plt.arrow(0,0.5,1,0, length_includes_head=True, width=0.001, head_width=0.02,color='k')
plt.arrow(0.5,0.7,0,-0.18, length_includes_head=True, width=0.001, head_width=0.02,color='k')
plt.vlines(0.5,0.49,0.51, color='k')
plt.text(0.5,0.72,'Performance Threshold',horizontalalignment='center')
plt.text(0,0.48,'0',verticalalignment='top')
plt.text(0.9,0.48,'Individual performance\nmeasure\n(likes, reputation points, etc.)\nas outcome of task 2',horizontalalignment='center',verticalalignment='top')
plt.text(0.5,0.48,r'$T$',horizontalalignment='center',verticalalignment='top')
plt.text(0.55,0.6,r'$\Delta$ task 1 $\uparrow$', color='r')
plt.axis('off')
plt.title('Delegation of authority on task 1')
plt.tight_layout()
# resultgraph_RF1.png

plt.figure()
plt.xlim([0,1])
plt.ylim([0.4,0.77])
plt.arrow(0,0.5,1,0, length_includes_head=True, width=0.001, head_width=0.02,color='k')
plt.arrow(0.5,0.7,0,-0.18, length_includes_head=True, width=0.001, head_width=0.02,color='k', alpha=0.1)
plt.vlines(0.5,0.49,0.51, color='k', alpha=0.1)
plt.text(0.5,0.72,'Performance Threshold',horizontalalignment='center', alpha=0.1)
#plt.arrow(0.7,0.7,0,-0.18, length_includes_head=True, width=0.001, head_width=0.02,color='k')
plt.vlines(0.7,0.49,0.51, color='k')
#plt.text(0.7,0.72,'New Threshold',horizontalalignment='center')
plt.text(0,0.48,'0',verticalalignment='top')
plt.text(0.9,0.48,'Individual performance\nmeasure\n(likes, reputation points, etc.)\nas outcome of task 2',horizontalalignment='center',verticalalignment='top')
plt.text(0.5,0.48,r'$T$',horizontalalignment='center',verticalalignment='top', alpha=0.1)
plt.text(0.55,0.6,r'$\Delta$ task 1 $\downarrow$', color='r')
plt.arrow(0.52,0.475,0.16,0, length_includes_head=True, width=0.001, head_width=0.01,color='k')
plt.text(0.7,0.48,r'$T^{\prime}$',horizontalalignment='center',verticalalignment='top')
plt.axis('off')
plt.title('Delegation of authority on task 1')
plt.tight_layout()
# resultgraph_RF2.png


plt.figure()
plt.xlim([0,1])
plt.ylim([0.4,0.77])
plt.arrow(0,0.5,1,0, length_includes_head=True, width=0.001, head_width=0.02,color='k')
plt.arrow(0.5,0.7,0,-0.18, length_includes_head=True, width=0.001, head_width=0.02,color='k')
plt.vlines(0.5,0.49,0.51, color='k')
plt.text(0.5,0.72,'Performance Threshold',horizontalalignment='center')
plt.text(0,0.48,'0',verticalalignment='top')
plt.text(0.9,0.48,'Reputation points\n(obtained answering)',horizontalalignment='center',verticalalignment='top')
plt.text(0.5,0.48,r'$2000$',horizontalalignment='center',verticalalignment='top')
plt.axis('off')
plt.title('Delegation of authority on Editing')
plt.tight_layout()
# delegation_design.png

plt.figure()
plt.xlim([0,1])
plt.ylim([0.4,0.77])
plt.arrow(0,0.5,1,0, length_includes_head=True, width=0.001, head_width=0.02,color='k')
plt.arrow(0.5,0.7,0,-0.18, length_includes_head=True, width=0.001, head_width=0.02,color='k', alpha=0.2)
plt.vlines(0.5,0.49,0.51, color='k', alpha=0.2)
plt.text(0.5,0.72,'Performance Threshold',horizontalalignment='center', alpha=0.2)
#plt.arrow(0.7,0.7,0,-0.18, length_includes_head=True, width=0.001, head_width=0.02,color='k')
plt.vlines(0.7,0.49,0.51, color='k')
#plt.text(0.7,0.72,'New Threshold',horizontalalignment='center')
plt.text(0,0.48,'0',verticalalignment='top')
plt.text(0.9,0.48,'Reputation points\n(obtained answering)',horizontalalignment='center',verticalalignment='top')
plt.text(0.5,0.48,r'$1000$',horizontalalignment='center',verticalalignment='top', alpha=0.1)
plt.text(0.55,0.6,'Some users\nloose authority', color='tab:orange')
plt.arrow(0.55,0.475,0.1,0, length_includes_head=True, width=0.001, head_width=0.01,color='k')
plt.text(0.7,0.48,r'$2000$',horizontalalignment='center',verticalalignment='top')
plt.axis('off')
plt.title('Delegation of authority on Editing')
plt.tight_layout()
# RF2specific.png
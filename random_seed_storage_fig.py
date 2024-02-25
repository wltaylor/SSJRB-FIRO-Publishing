#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 11:05:33 2023

@author: williamtaylor
"""

import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import json
import model
import xarray as xr
import datetime
from util import *

# compare FIRO and baseline simulations
# we have data since 2013-11-01, but those years are very dry
sd = '2013-11-01'
nodes = json.load(open('data/nodes.json'))
df_hist = pd.read_csv('data/historical.csv', index_col=0, parse_dates=True)[sd:]
medians = pd.read_csv('data/historical_medians.csv', index_col=0)

# the non-firo params have already been calibrated in fit_historical.py
# 4/26/23 the fitted TOCS parameters are close to CALFEWS values
params = json.load(open('data/params.json'))

rk = [k for k in nodes.keys() if (nodes[k]['type'] == 'reservoir') and (nodes[k]['fit_policy'])]
Kr = np.array([nodes[k]['capacity_taf'] * 1000 for k in rk])
max_release = np.array([nodes[k]['safe_release_cfs'] for k in rk])
ramping_rate = np.array([nodes[k]['ramping_rate'] for k in rk])
pk = [k for k in nodes.keys() if (nodes[k]['type'] == 'pump') and (nodes[k]['fit_policy'])]
Kp = np.array([nodes[k]['capacity_cfs'] for k in pk])
input_data = get_simulation_data(rk, pk, df_hist, medians, init_storage=True)


Qf = xr.open_dataset('data/cnrfc14d.nc')['Qf']
# <xarray.DataArray 'Qf' (site, time, ens, lead)>
Qf = Qf.values[:,:,:40,:] # numpy array, keep 40 ens members as limiting value
Qf.sort(axis=2) # should help with runtime later - pre-sort the ensemble
#Qf = xr.open_dataset('data/perfect_forecast.nc')['inflow']
#Qf = Qf.values
names = ['SHA','ORO','BUL','FOL','PAR','NHG','NML','DNP','EXC','MIL','PNF','TRM','SCC','ISB']

capacity = np.zeros(len(names))
for i in range(len(capacity)):
    capacity[i] = nodes[names[i]]['capacity_taf']

#%% Run all 10 seeds in a loop
plt.rcParams['figure.figsize'] = [8, 6]
for j in range(10):
    file_name = f"params_{j}.csv" # read in the 10 different seed results
    params_seed = pd.read_csv(file_name, header=None)
    params_seed['Names'] = names
    params_seed = params_seed.set_index('Names')
    params = json.load(open('data/params_firo.json'))
    for i,r in enumerate(rk): # replace the FIRO values in params with the seed results
        params[r][-3] = params_seed.loc[r,0] #days to allow zero risk
        params[r][-2] = params_seed.loc[r,1] #slope of risk curve
        params[r][-1] = params_seed.loc[r,2] #tocs multiplier
    
    # run simulation
    params_in = tuple(np.array(v) for k,v in params.items())
    results = model.simulate(params_in, Kr, Kp, *input_data, max_release, ramping_rate, use_firo=True, Qf=Qf)
    df_opt = results_to_df(results, index=df_hist.index, res_keys=rk)
    
    # plot results
    for i in range(len(names)):
        res = names[i]
        if j == 8:
            plt.scatter(res, df_opt.loc[:,res+'_storage_af'].mean()/capacity[i]/1000, label = 'Seed 8', c = 'r', marker = '*', s=75)
        else:
            plt.scatter(res, df_opt.loc[:,res+'_storage_af'].mean()/capacity[i]/1000, label = 'All other seeds', c = 'black', marker = ".", s = 50, alpha = 0.5)
plt.xlabel('Reservoir')
plt.ylabel('Normalized Capacity (unitless)')
legend_handles = [
    Line2D([0], [0], marker='o', color='w', label='Seed 0 (J = 0.579)', markerfacecolor='black', markersize=8),
    Line2D([0], [0], marker='o', color='w', label='Seed 1 (J = 0.585)', markerfacecolor='black', markersize=8),
    Line2D([0], [0], marker='o', color='w', label='Seed 2 (J = 0.584)', markerfacecolor='black', markersize=8),
    Line2D([0], [0], marker='o', color='w', label='Seed 3 (J = 0.582)', markerfacecolor='black', markersize=8),
    Line2D([0], [0], marker='o', color='w', label='Seed 4 (J = 0.583)', markerfacecolor='black', markersize=8),
    Line2D([0], [0], marker='o', color='w', label='Seed 5 (J = 0.581)', markerfacecolor='black', markersize=8),
    Line2D([0], [0], marker='o', color='w', label='Seed 6 (J = 0.584)', markerfacecolor='black', markersize=8),
    Line2D([0], [0], marker='o', color='w', label='Seed 7 (J = 0.584)', markerfacecolor='black', markersize=8),
    Line2D([0], [0], marker='*', color='w', label='Seed 8 (J = 0.586)', markerfacecolor='red', markersize=12),
    Line2D([0], [0], marker='o', color='w', label='Seed 9 (J = 0.583)', markerfacecolor='black', markersize=8),
]

plt.legend(handles=legend_handles)

plt.title('Normalized Storage by Reservoir')

plt.subplot

#%%

import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec

# useful for determining which axes is which (not used)
#def annotate_axes(fig):
#    for i, ax in enumerate(fig.axes):
#        ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
#        ax.tick_params(labelbottom=False, labelleft=False)

plt.rcParams['figure.figsize'] = [12, 6]

fig = plt.figure()
fig.suptitle("Random seed simulation results", fontsize=16)

# define first gridspace and axes (left)
# simulate model for each seed and calculate J-value
gs1 = GridSpec(1, 1, left = 0.05, right = 0.48)
ax1 = fig.add_subplot(gs1[0])

for j in range(10):
    file_name = f"params_{j}.csv" # read in the 10 different seed results
    params_seed = pd.read_csv(file_name, header=None)
    params_seed['Names'] = names
    params_seed = params_seed.set_index('Names')
    params = json.load(open('data/params_firo.json'))
    for i,r in enumerate(rk): # replace the FIRO values in params with the seed results
        params[r][-3] = params_seed.loc[r,0] #days to allow zero risk
        params[r][-2] = params_seed.loc[r,1] #slope of risk curve
        params[r][-1] = params_seed.loc[r,2] #tocs multiplier
    
    # run simulation
    params_in = tuple(np.array(v) for k,v in params.items())
    results = model.simulate(params_in, Kr, Kp, *input_data, max_release, ramping_rate, use_firo=True, Qf=Qf)
    df_opt = results_to_df(results, index=df_hist.index, res_keys=rk)
    
    # plot results
    for i in range(len(names)):
        res = names[i]
        if j == 8:
            ax1.scatter(res, df_opt.loc[:,res+'_storage_af'].mean()/capacity[i]/1000, label = 'Seed 8', c = 'r', marker = '*', s=75)
        else:
            ax1.scatter(res, df_opt.loc[:,res+'_storage_af'].mean()/capacity[i]/1000, label = 'All other seeds', c = 'black', marker = ".", s = 50, alpha = 0.5)
plt.xlabel('Reservoir')
plt.ylabel('Normalized Capacity (unitless)')
legend_handles = [
    Line2D([0], [0], marker='o', color='w', label='Seed 0 (J = 0.582)', markerfacecolor='black', markersize=8),
    Line2D([0], [0], marker='o', color='w', label='Seed 1 (J = 0.581)', markerfacecolor='black', markersize=8),
    Line2D([0], [0], marker='o', color='w', label='Seed 2 (J = 0.583)', markerfacecolor='black', markersize=8),
    Line2D([0], [0], marker='o', color='w', label='Seed 3 (J = 0.583)', markerfacecolor='black', markersize=8),
    Line2D([0], [0], marker='*', color='w', label='Seed 4 (J = 0.585)', markerfacecolor='red', markersize=12),
    Line2D([0], [0], marker='o', color='w', label='Seed 5 (J = 0.579)', markerfacecolor='black', markersize=8),
    Line2D([0], [0], marker='o', color='w', label='Seed 6 (J = 0.582)', markerfacecolor='black', markersize=8),
    Line2D([0], [0], marker='o', color='w', label='Seed 7 (J = 0.581)', markerfacecolor='black', markersize=8),
    Line2D([0], [0], marker='o', color='w', label='Seed 8 (J = 0.578)', markerfacecolor='black', markersize=8),
    Line2D([0], [0], marker='o', color='w', label='Seed 9 (J = 0.582)', markerfacecolor='black', markersize=8),
]

plt.legend(handles=legend_handles, loc='lower left')

plt.title('Normalized storage by reservoir')

# define second gridspace and axes (right)
# read in parameter values and plot
gs2 = GridSpec(3, 1, left = 0.54, right = 0.98, hspace = 0.45)
opt_params = pd.read_csv('data/optimized_params.csv')
opt_params = opt_params.drop(['Unnamed: 0'], axis=1)
params0 = pd.read_csv('params_0.csv', header=None)
params1 = pd.read_csv('params_1.csv', header=None)
params2 = pd.read_csv('params_2.csv', header=None)
params3 = pd.read_csv('params_3.csv', header=None)
params4 = pd.read_csv('params_4.csv', header=None)
params5 = pd.read_csv('params_5.csv', header=None)
params6 = pd.read_csv('params_6.csv', header=None)
params7 = pd.read_csv('params_7.csv', header=None)
params8 = pd.read_csv('params_8.csv', header=None)
params9 = pd.read_csv('params_9.csv', header=None)

ax2 = fig.add_subplot(gs2[0])
plt.scatter(opt_params.iloc[:,0], params0.iloc[:,0], c = 'black', alpha = 0.3)
plt.scatter(opt_params.iloc[:,0], params1.iloc[:,0], c = 'black', alpha = 0.3)
plt.scatter(opt_params.iloc[:,0], params2.iloc[:,0], c = 'black', alpha = 0.3)
plt.scatter(opt_params.iloc[:,0], params3.iloc[:,0], c = 'black', alpha = 0.3)
plt.scatter(opt_params.iloc[:,0], params4.iloc[:,0], c = 'red', alpha = 1, marker = '*', s = 30)
plt.scatter(opt_params.iloc[:,0], params5.iloc[:,0], c = 'black', alpha = 0.3)
plt.scatter(opt_params.iloc[:,0], params6.iloc[:,0], c = 'black', alpha = 0.3)
plt.scatter(opt_params.iloc[:,0], params7.iloc[:,0], c = 'black', alpha = 0.3)
plt.scatter(opt_params.iloc[:,0], params8.iloc[:,0], c = 'black', alpha = 0.3)
plt.scatter(opt_params.iloc[:,0], params9.iloc[:,0], c = 'black', alpha = 0.3)
plt.title('Days without risk')

ax3 = fig.add_subplot(gs2[1])
plt.scatter(opt_params.iloc[:,0], params0.iloc[:,1], c = 'black', alpha = 0.3)
plt.scatter(opt_params.iloc[:,0], params1.iloc[:,1], c = 'black', alpha = 0.3)
plt.scatter(opt_params.iloc[:,0], params2.iloc[:,1], c = 'black', alpha = 0.3)
plt.scatter(opt_params.iloc[:,0], params3.iloc[:,1], c = 'black', alpha = 0.3)
plt.scatter(opt_params.iloc[:,0], params4.iloc[:,1], c = 'red', alpha = 1, marker='*', s = 30)
plt.scatter(opt_params.iloc[:,0], params5.iloc[:,1], c = 'black', alpha = 0.3)
plt.scatter(opt_params.iloc[:,0], params6.iloc[:,1], c = 'black', alpha = 0.3)
plt.scatter(opt_params.iloc[:,0], params7.iloc[:,1], c = 'black', alpha = 0.3)
plt.scatter(opt_params.iloc[:,0], params8.iloc[:,1], c = 'black', alpha = 0.3)
plt.scatter(opt_params.iloc[:,0], params9.iloc[:,1], c = 'black', alpha = 0.3)
plt.title('Slope of risk curve')

ax4 = fig.add_subplot(gs2[2])
plt.scatter(opt_params.iloc[:,0], params0.iloc[:,2], c = 'black', alpha = 0.3)
plt.scatter(opt_params.iloc[:,0], params1.iloc[:,2], c = 'black', alpha = 0.3)
plt.scatter(opt_params.iloc[:,0], params2.iloc[:,2], c = 'black', alpha = 0.3)
plt.scatter(opt_params.iloc[:,0], params3.iloc[:,2], c = 'black', alpha = 0.3)
plt.scatter(opt_params.iloc[:,0], params4.iloc[:,2], c = 'red', alpha = 1, marker = '*', s = 30)
plt.scatter(opt_params.iloc[:,0], params5.iloc[:,2], c = 'black', alpha = 0.3)
plt.scatter(opt_params.iloc[:,0], params6.iloc[:,2], c = 'black', alpha = 0.3)
plt.scatter(opt_params.iloc[:,0], params7.iloc[:,2], c = 'black', alpha = 0.3)
plt.scatter(opt_params.iloc[:,0], params8.iloc[:,2], c = 'black', alpha = 0.3)
plt.scatter(opt_params.iloc[:,0], params9.iloc[:,2], c = 'black', alpha = 0.3)
plt.title('TOCS multiplier')






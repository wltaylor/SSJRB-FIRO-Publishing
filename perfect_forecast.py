#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 13:52:13 2023

@author: williamtaylor
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import netCDF4 as nc
#%%
df_hist = pd.read_csv('data/historical.csv', index_col=0, parse_dates=True)
df_hist = df_hist['2013-11-1':]

names = ['SHA','ORO','BUL','FOL','PAR','NHG','NML','DNP','EXC','MIL','PNF','TRM','SCC','ISB']
inflows = np.array(df_hist[['SHA_inflow_cfs','ORO_inflow_cfs','BUL_inflow_cfs','FOL_inflow_cfs','PAR_inflow_cfs','NHG_inflow_cfs','NML_inflow_cfs','DNP_inflow_cfs','EXC_inflow_cfs','MIL_inflow_cfs', 'PNF_inflow_cfs', 'TRM_inflow_cfs', 'SCC_inflow_cfs', 'ISB_inflow_cfs']])

perfect = np.zeros((14,len(inflows), 40, 14)) #site, date, ensembles, lead time

for i in range(len(names)):
    for j in range(len(inflows)):
        for k in range(0,40):
            for l in range(0,14):
                day = j+l
                
                if j < 3478:
                    perfect[i,j,k,l] = inflows[day,i]
                else:
                    perfect[i,j,k,l] = inflows[3477,i]
                
#%%
plt.plot(perfect[1,0,:,:].T)
#plt.plot(perfect[0,3478,:,:].T)
plt.plot(df_hist['2013-11-01':'2013-11-15']['ORO_inflow_cfs'].values)
plt.legend(['Perfect','Observed'])
#%% Save as a netcdf file
file = nc.Dataset('data/perfect_forecast.nc', 'w', format = 'NETCDF4')

site = file.createDimension('site', 14)
time = file.createDimension('time', len(inflows))
ensemble = file.createDimension('ensemble', 40)
lead = file.createDimension('lead', 14)

inflow_var = file.createVariable('inflow', 'float64', ('site', 'time', 'ensemble', 'lead'))


inflow_var[:,:,:,:] = perfect[:,:,:,:]

#%% test
perfect_forecast = xr.open_dataset('data/perfect_forecast.nc')['inflow']
perfect_forecast = perfect_forecast.values

plt.plot(perfect_forecast[0,0,:,:].T)


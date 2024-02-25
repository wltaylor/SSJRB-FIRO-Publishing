#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 14:09:35 2023

@author: williamtaylor
"""

import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import datetime
from datetime import timedelta
from scipy import stats
import os
import plotnine as p9
from plotnine import ggplot, geom_point, aes, stat_smooth, facet_wrap, geom_segment
import xarray as xr
import json
import matplotlib
from matplotlib import cm
import matplotlib.colors as colors
from matplotlib.cm import ScalarMappable

#%% Importing data

Rbaseline = pd.read_csv('data/baseline_release.csv', header = None)
Roptbase = pd.read_csv('data/optimized_baseline_release.csv', header = None)
Ropt = pd.read_csv('data/optimized_release.csv', header = None)
Rperf = pd.read_csv('data/perfect_release.csv', header = None)

Sbaseline = pd.read_csv('data/baseline_storage.csv', header = None)
Soptbase = pd.read_csv('data/optimized_baseline_storage.csv', header = None)
Sopt = pd.read_csv('data/optimized_storage.csv', header = None)
Sperf = pd.read_csv('data/perfect_storage.csv', header = None)

df_hist = pd.read_csv('data/historical.csv', index_col=0, parse_dates=True)
df_hist = df_hist['2013-11-1':]

nodes = json.load(open('data/nodes.json'))

#%% Preprocessing

# initializing date
start_date = datetime.datetime.strptime('11-01-2013', "%m-%d-%Y")

# initializing K
K = len(Sbaseline)
dates = pd.date_range(start_date, periods=K)

names = ['SHA','ORO','BUL','FOL','PAR','NHG','NML','DNP','EXC','MIL','PNF','TRM','SCC','ISB']

Sbaseline = pd.DataFrame(data = Sbaseline.values,
                  index = dates,
                  columns = names)

Rbaseline = pd.DataFrame(data = Rbaseline.values,
                  index = dates,
                  columns = names)

Roptbase = pd.DataFrame(data = Roptbase.values,
                  index = dates,
                  columns = names)

Soptbase = pd.DataFrame(data = Soptbase.values,
                  index = dates,
                  columns = names)

Ropt = pd.DataFrame(data = Ropt.values,
                  index = dates,
                  columns = names)

Sopt = pd.DataFrame(data = Sopt.values,
                  index = dates,
                  columns = names)

Rperf = pd.DataFrame(data = Rperf.values,
                  index = dates,
                  columns = names)

Sperf = pd.DataFrame(data = Sperf.values,
                  index = dates,
                  columns = names)

#%% Figure 1 - Location Overview

import cartopy
from cartopy import crs
from cartopy import feature as cfeature

fig = plt.figure(figsize = (16,6))

ax = fig.add_subplot(1,2,1, projection = crs.PlateCarree())

ax.set_extent([-124, -117, 35, 42],
              crs=crs.PlateCarree()) ## Important

ax.add_feature(cfeature.COASTLINE, rasterized=True)
ax.add_feature(cfeature.LAND, color="cornsilk", alpha=1.0, rasterized=True)
ax.add_feature(cfeature.BORDERS, linestyle="--", rasterized=True)
ax.add_feature(cfeature.OCEAN, color="skyblue", alpha=0.4, rasterized=True)
ax.add_feature(cfeature.RIVERS, edgecolor="skyblue", rasterized=True)
ax.add_feature(cfeature.STATES, rasterized=True)
#plt.title("Sacramento, San Joaquin and Tulare Reservoirs")

for k in nodes:
    if nodes[k]['type'] == 'reservoir':
        lon, lat = nodes[k]['coords']

        if k not in ('CMN', 'SLF', 'LUS'):
            K = nodes[k]['capacity_taf']
            #plot reservoir markers
            ax.scatter(lon, lat,
                       s = 3*np.sqrt(K), 
                       marker = '^', 
                       alpha = 0.5, 
                       zorder = 10, 
                       c = 'k', 
                       edgecolor = 'black', 
                       transform = crs.PlateCarree())
            #plot reservoir text name
            ax.annotate(k, 
                        (lon+0.13, lat-0.01), 
                        transform = crs.PlateCarree())
#plot legend
for capacities in [500, 2500, 4500]:
    plt.scatter([], [], c='k',
                alpha=0.5, 
                s=3*np.sqrt(capacities),
                label=capacities,
                marker = '^')
plt.legend(scatterpoints=1, frameon=False, labelspacing=1, title='Reservoir Capacity', facecolor='white')

plt.show()

#%% Figure 2 - Example forecast

from datetime import date
plt.rcParams['figure.figsize'] = [12, 6]

nodes = json.load(open('data/nodes.json'))
reservoirs = [k for k in nodes.keys() if nodes[k]['type'] == 'reservoir']
cnrfc_ids = [nodes[r].get('CNRFC_ID') for r in reservoirs]
sites = [c for c in cnrfc_ids if c is not None]
print(sites)

sd = '2013-11-01' # cnrfc start date
lead_days = 14
max_ens = 60
today = date.today().strftime("%Y-%m-%d")
site = 'ORDC1'

ds = xr.open_dataset('data/cnrfc14d.nc')

Qo = ds['Qo'].loc[site, '2017-02-01':'2017-02-13']

base_date = '2017-02-01'
Qf = ds['Qf'].loc[site, base_date, :, :]
fcst_dates = [pd.to_datetime(base_date) + pd.DateOffset(days=int(d-1)) for d in Qf.lead.values]
plt.subplot(1,2,1)
Qo.plot(color='k', zorder=5)
plt.plot(fcst_dates, Qf.T, color='0.7')
plt.ylabel('Q (cfs)')
plt.xlabel('Lead time (days)')
plt.legend(['Obs', 'Fcst'])
plt.tight_layout()
plt.title('1 Feb 2017 Forecasted Inflow (ORO)')

# Nash-Sutcliffe Efficiency

def nse(predictions, targets):
  return (1-(np.sum((targets-predictions)**2)/np.sum((targets-np.mean(targets))**2)))

path = '../ssjrb-firo/'
nodes = json.load(open('data/nodes.json'))
rk = [k for k in nodes.keys() if (nodes[k]['type'] == 'reservoir') and (nodes[k]['fit_policy'])]

# forecast data
ds = xr.open_dataset('data/cnrfc14d.nc')
#Qf = xr.open_dataset('data/perfect_forecast.nc')['inflow']
Qf = ds['Qf'] # <xarray.DataArray 'Qf' (site, time, ens, lead)>
Qo = ds['Qo'] # <xarray.DataArray 'Qf' (site, time)>


Qo = Qo.values
Qf_ens_mean = np.nanmean(Qf.values, axis=2) # ensemble mean, dimensions (site, time, lead)
Qf_ens_max = np.nanmax(Qf.values, axis=2) # ensemble max, dimensions (site, time, lead)
# using max for now instead of 90th pctile - function nanquantile is very slow for large array

nses = np.zeros((14,14))
FPRs = np.zeros((14,14))
FNRs = np.zeros((14,14))

for site in range(14):
  for lead in range(14):

    if lead == 0:
      pred = Qf_ens_mean[site,:,lead]
      pred_max = Qf_ens_max[site,:,lead]
      obs = Qo[site,:]
    else:
      pred = Qf_ens_mean[site,:-lead,lead]
      pred_max = Qf_ens_max[site,:-lead,lead]
      obs = Qo[site,lead:]

    ix = (~np.isnan(pred) & ~np.isnan(pred_max) & ~np.isnan(obs))
    pred = pred[ix] # only keep non-nan values
    pred_max = pred_max[ix]
    obs = obs[ix]

    # if rk[site] == 'MIL' and lead==0:
    #   plt.plot(obs)
    #   plt.plot(pred)
    #   plt.show()
    #   plt.scatter(obs,pred)
    #   plt.plot([0,obs.max()], [0,obs.max()], c='red')
    #   plt.show()

    nses[site,lead] = nse(pred,obs)

    # false positive: prediction above 90th pctile, but observed is not
    FP = ((pred_max) > np.quantile(obs,0.9)) & (obs < np.quantile(obs,0.9))
    FPRs[site,lead] = FP.sum() / len(FP)
    # false negative: prediction below 90th pctile, but observed is above
    FN = ((pred_max) < np.quantile(obs,0.9)) & (obs > np.quantile(obs,0.9))
    FNRs[site,lead] = FN.sum() / len(FN)

names = ['SHA','ORO','BUL','FOL','PAR','NHG','NML','DNP','EXC','MIL','PNF','TRM','SCC','ISB']


plt.subplot(1,2,2)
for k in range(len(names)):
    name = names[k]
    x = np.zeros(14)
    for i in range(len(x)):
        x[i] = nses[k,i]
    plt.plot(x)
    plt.annotate(name, (10, x[10]))
plt.legend(names)
#plt.legend(names, bbox_to_anchor=(1.12,1.0), loc = 'upper right')
plt.ylabel('NSE')
plt.xlabel('Lead time (days)')
plt.title('Nash Sutcliffe Efficiency')

# plt.subplot(3,1,2)
# plt.plot(range(1,15), FPRs.T)
# plt.legend(rk, bbox_to_anchor = (1.02,1))
# plt.xlabel('Lead time (days)')
# plt.ylabel('FPR')
# plt.title('False Positive Rate')

# plt.subplot(3,1,3)
# plt.plot(range(1,15), FNRs.T)
# plt.xlabel('Lead time (days)')
# plt.ylabel('FNR')
# plt.title('False Negative Rate')

plt.tight_layout()
plt.show()

#%% Figure 4 - Average Storage per Policy
plt.rcParams['figure.figsize'] = [8, 6]

capacity = np.zeros(len(names))
for i in range(len(capacity)):
    capacity[i] = nodes[names[i]]['capacity_taf']

for i in range(len(names)):
    res = names[i]
    plt.scatter(res, Sbaseline.iloc[:,i].mean()/capacity[i]/1000, label = 'Baseline', color = 'r', alpha = 0.5)
    plt.scatter(res, Soptbase.iloc[:,i].mean()/capacity[i]/1000, label = 'Baseline-TOCS', color = 'b', alpha = 0.5)
    plt.scatter(res, Sopt.iloc[:,i].mean()/capacity[i]/1000, label = 'Forecast', color = 'g', alpha = 0.5)
    plt.scatter(res, Sperf.iloc[:,i].mean()/capacity[i]/1000, label = 'Perfect', color = 'c', alpha = 0.5)
plt.xlabel('Reservoir')
plt.ylabel('Normalized Capacity (unitless)')
plt.legend(['Baseline (J = 0.490)','Baseline-TOCS (J = 0.571)', 'Forecast (J = 0.585)', 'Perfect (J = 0.587)'])
plt.title('Normalized Storage by Reservoir')
plt.show()

#%% absolute storage difference (af)
total = 0
for i in range(len(names)):
    total += Sopt.iloc[:,i].mean() - Sbaseline.iloc[:,i].mean()
print(total)
#%% Figure 5 - Storage and Release by Reservoir

#show Oroville during 2017 spill
df_hist = pd.read_csv('data/historical.csv', index_col=0, parse_dates=True)
df_hist = df_hist['2013-11-1':]

from datetime import datetime
plt.rcParams['figure.figsize'] = [9, 4]
startdate = datetime(2017,1,15)
enddate = datetime(2017,2,28)
i = 1
plt.subplot(211)
plt.plot(Sbaseline.iloc[:,i],'r')
plt.plot(Soptbase.iloc[:,i],'b')
plt.plot(Sopt.iloc[:,i],'g')
plt.plot(Sperf.iloc[:,i],'c')
plt.plot(df_hist[names[i]+"_storage_af"], 'black')
#plt.xlim(startdate,enddate)
plt.ylabel("Storage (AF)")
plt.suptitle(names[i], fontsize = 16)
#plt.legend(["Baseline", "Baseline Optimized", "Optimized", "Perfect"], bbox_to_anchor=(1.35,-.10), loc = 'center right')
    
plt.subplot(212)
plt.plot(Rbaseline.iloc[:,i],'r')
plt.plot(Roptbase.iloc[:,i],'b')
plt.plot(Ropt.iloc[:,i],'g')
plt.plot(Rperf.iloc[:,i],'c')
plt.plot(df_hist[names[i]+"_inflow_cfs"], 'black')
plt.xlabel("Date")
plt.ylabel("Release (cfs)")
#plt.legend(["Baseline", "Forecast", "Observed Release"])
plt.legend(["Baseline", "Baseline-TOCS", "Forecast", "Perfect","Observed"])
plt.xlim(startdate,enddate)
plt.xticks(rotation=45)
plt.show()

#%% Figure 6 - Scatter Plot of Parameter Values

# show parameter values for P1, P2, P3. P3 (TOCS multiplier) will be the only one with multiple points

#import all the parameters
plt.rcParams['figure.figsize'] = [8, 6]

baseline_params = np.ones(14)
baseopt_params = pd.read_csv('data/optimized_baseline_params.csv', header=None)
opt_params = pd.read_csv('data/optimized_params.csv')
opt_params = opt_params.drop(['Unnamed: 0'], axis=1)
perf_params = pd.read_csv('data/perfect_params.csv')
perf_params = perf_params.drop(['Unnamed: 0'], axis=1)

plt.subplot(3,1,1)
plt.scatter(opt_params.iloc[:,0], opt_params.iloc[:,1], color = 'g')
plt.title('Lead days with no allowable risk')
plt.ylim(0,10)

plt.subplot(3,1,2)
plt.scatter(opt_params.iloc[:,0], opt_params.iloc[:,2], color = 'g')
plt.ylim(0,0.1)
plt.title('Slope of risk curve')

plt.subplot(3,1,3)
plt.scatter(opt_params.iloc[:,0], baseline_params, color = 'r')
plt.scatter(opt_params.iloc[:,0], baseopt_params, color = 'b')
plt.scatter(opt_params.iloc[:,0], opt_params.iloc[:,3], color = 'g')
plt.scatter(perf_params.iloc[:,0], perf_params.iloc[:,3], color = 'c')
plt.legend(['Baseline','Baseline-TOCS','Forecast','Perfect'], loc='lower right')
plt.ylim(1.0,2.0)
plt.title('TOCS Multiplier')

plt.tight_layout()
plt.show()

#%% Scatter plot by random seed

params_files = [f'data/params_{i}.csv' for i in range(10)]
params_data = [pd.read_csv(file, header=None) for file in params_files]

opt_params = pd.read_csv('data/optimized_params.csv')
opt_params = opt_params.drop(['Unnamed: 0'], axis=1)
fig, axes = plt.subplots(3,1, figsize=(8,10))

titles = ['Days without risk', 'Slope of risk curve','TOCS multiplier']

for i, ax in enumerate(axes):
    for j, params in enumerate(params_data):
        color = 'red' if j == 4 else 'black'
        alpha = 1 if j == 4 else 0.5
        marker = '*' if j == 4 else 'o'
        ax.scatter(opt_params.iloc[:,0], params.iloc[:,i], c=color, alpha=alpha, marker = marker)
    ax.set_ylim([(0, 10), (0, 0.1), (1.0, 2.0)][i])
    ax.set_title(titles[i])

# Adjust layout
plt.tight_layout()
plt.show()

#%%
color_variable = np.zeros(14)
for i in range(14):
    
    color_variable[i] = opt_params.iloc[i,3]
    
cmap = plt.cm.get_cmap('viridis')  # Choose a colormap
norm = plt.Normalize(vmin=min(color_variable), vmax=max(color_variable))
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
# optimized risk curves by res
for k in range(len(names)):
    res = k
    piecewise = np.zeros(14)
    for i in range(len(piecewise)):
        if i < opt_params.iloc[res,1]:
            piecewise[i] = 0
        else:
            piecewise[i] = (i - np.round(opt_params.iloc[res,1]))*opt_params.iloc[res,2]
    plt.plot(piecewise, c=cmap(norm(color_variable[k])), label=names[k])
#plt.legend()
cbar = plt.colorbar(sm)
cbar.set_label('TOCS multiplier')
plt.xlabel('Lead day')
plt.ylabel('Risk percent')
plt.title('Optimal Risk Curve by Reservoir')
plt.show()

#%% Figure 7 - Comparison of max releases by policy

#policies = ['Baseline','Baseline Optimized','Optimized','Perfect']
policies = ['Baseline','Baseline-TOCS','Forecast','Perfect']
names = ['SHA','ORO','BUL','FOL','PAR','NHG','NML','DNP','EXC','MIL','PNF','TRM','SCC','ISB']

maximums = np.zeros((4,len(names)))

for i in range(len(names)):
    maximums[0,i] = np.max(Rbaseline.iloc[:,i]) #baseline 
    maximums[1,i] = np.max(Ropt.iloc[:,i]) #baseline optimized
    maximums[2,i] = np.max(Ropt.iloc[:,i]) # optimized
    maximums[3,i] = np.max(Rperf.iloc[:,i]) #perfect

maximums_df = pd.DataFrame(data = maximums.T,
                           index = names,
                           columns = policies)

segments = np.zeros((14,2))
for i in range(len(segments)):
    segments[i,0] = np.min(maximums_df.iloc[i,:])
    segments[i,1] = np.max(maximums_df.iloc[i,:])

segments_df = pd.DataFrame(data = segments,
                           index = names,
                           columns = ['Min','Max'])
segments_df = segments_df.reset_index()
segments_df = segments_df.rename(columns={"index":"Reservoir"})

maximums_df = maximums_df.reset_index()
maximums_df = maximums_df.rename(columns={'index':'Reservoir'})

points_df = pd.DataFrame(columns = ['Reservoir','Policy','Release'])
for j in range(len(policies)):
    for i in range(len(names)):
        points_df.loc[len(points_df.index)] = [names[i],policies[j],maximums_df.iloc[i,j+1]]
        
fig1 = (ggplot()
  #Range strip
  +geom_segment(
      segments_df,
      aes(x='Min', xend='Max', y='Reservoir', yend='Reservoir'),
      size = 6,
      color='#a7a9ac'
      )
  #Policy max releases
  +geom_point(
      points_df,
      aes('Release','Reservoir',color='Policy'),
      size = 5,
      stroke=0.7,
      )
  + p9.ylab('Reservoir')
  + p9.xlab('Maximum Release (cfs)')
  + p9.ggtitle('Change in maximum release by policy')
  )
print(fig1)

nodes = json.load(open('data/nodes.json'))

safe_release = np.zeros(len(names))
for i in range(len(names)):
    safe_release[i] = nodes[names[i]]['safe_release_cfs']

#normalize the segments data
for i in range(len(names)):
    segments_df['Min'][i] = segments_df['Min'][i]/safe_release[i]
    segments_df['Max'][i] = segments_df['Max'][i]/safe_release[i]

#normalize the points data
points_df = pd.DataFrame(columns = ['Reservoir','Policy','Release'])
for j in range(len(policies)):
    for i in range(len(names)):
        points_df.loc[len(points_df.index)] = [names[i],policies[j],maximums_df.iloc[i,j+1]/safe_release[i]]    

fig2 = (ggplot()
  #Range strip
  +geom_segment(
      segments_df,
      aes(x='Min', xend='Max', y='Reservoir', yend='Reservoir'),
      size = 6,
      color='#a7a9ac'
      )
  #Policy max releases
   +geom_point(
       points_df,
       aes('Release','Reservoir',color='Policy'),
       size = 5,
       stroke=0.7,
       )
  + p9.ylab('Reservoir')
  + p9.xlab('Maximum Release (cfs/cfs)')
  + p9.ggtitle('Change in normalized maximum release by policy')
  )
print(fig2)

#%% Count the number of times the max release is hit for each policy

max_release_count = np.zeros((14,4))
tolerance = 0.5
for i in range(len(names)):
    max_release_count[i,0] = ((Rbaseline >= safe_release[i]-tolerance) & (Rbaseline <= safe_release[i] + tolerance)).sum().sum()
for i in range(len(names)):
    max_release_count[i,1] = ((Roptbase >= safe_release[i]-tolerance) & (Roptbase <= safe_release[i] + tolerance)).sum().sum()
for i in range(len(names)):
    max_release_count[i,2] = ((Ropt >= safe_release[i]-tolerance) & (Ropt <= safe_release[i] + tolerance)).sum().sum()
for i in range(len(names)):
    max_release_count[i,3] = ((Rperf >= safe_release[i]-tolerance) & (Rperf <= safe_release[i] + tolerance)).sum().sum()

#combine into one df
max_release_df = pd.DataFrame({
    'Baseline': max_release_count[:,0],
    'Baseline-Optimized': max_release_count[:,1],
    'Optimized': max_release_count[:,2],
    'Perfect': max_release_count[:,3]})
max_release_df.index = names

ax = max_release_df.plot(kind='bar', width=0.5, position=1, align='center')
plt.xlabel('Reservoir')
plt.ylabel('Count')
plt.title('Frequency of maximum release by policy')

#%% Count the number of times the max release is hit for each policy, drawdown period exlcuded
max_release_freq = np.zeros((14,4))

start_year = 2013
end_year = 2023

params = json.load(open('data/params.json'))

# baseline
for i in range(len(names)):          
    print(names[i])
    name = names[i]
    remaining_dataframes = []
    for year in range(start_year, end_year+1):
        drawdown = params[name][1] #determine the dates we need excluded for 'name' reservoir
        start_exclude = pd.to_datetime(f"{year}-10-01")
        end_exclude = start_exclude + timedelta(days=drawdown)
        excluded_date_ranges = [(start_exclude, end_exclude)] # list of our excluded dates
        year_data = Rbaseline[name][str(year)] # subset the release data by year and apply the exclusion period to each
        for start_date, end_date in excluded_date_ranges:
            excluded_range = year_data[(year_data.index < start_date) | (year_data.index > end_date)]
            year_data = excluded_range
        remaining_dataframes.append(year_data) # put all the data we want in the same dataframe
        
    combined_dataframe = pd.concat(remaining_dataframes)
    
    max_release_freq[i,0] = ((combined_dataframe >= safe_release[i]-tolerance) & (combined_dataframe <= safe_release[i] + tolerance)).sum().sum()

#baseline-optimized
for i in range(len(names)):          
    print(names[i])
    name = names[i]
    remaining_dataframes = []
    for year in range(start_year, end_year+1):
        drawdown = params[name][1] #determine the dates we need excluded for 'name' reservoir
        start_exclude = pd.to_datetime(f"{year}-10-01")
        end_exclude = start_exclude + timedelta(days=drawdown)
        excluded_date_ranges = [(start_exclude, end_exclude)] # list of our excluded dates
        year_data = Roptbase[name][str(year)] # subset the release data by year and apply the exclusion period to each
        for start_date, end_date in excluded_date_ranges:
            excluded_range = year_data[(year_data.index < start_date) | (year_data.index > end_date)]
            year_data = excluded_range
        remaining_dataframes.append(year_data) # put all the data we want in the same dataframe
        
    combined_dataframe = pd.concat(remaining_dataframes)
    
    max_release_freq[i,1] = ((combined_dataframe >= safe_release[i]-tolerance) & (combined_dataframe <= safe_release[i] + tolerance)).sum().sum()

#optimized
for i in range(len(names)):          
    print(names[i])
    name = names[i]
    remaining_dataframes = []
    for year in range(start_year, end_year+1):
        drawdown = params[name][1] #determine the dates we need excluded for 'name' reservoir
        start_exclude = pd.to_datetime(f"{year}-10-01")
        end_exclude = start_exclude + timedelta(days=drawdown)
        excluded_date_ranges = [(start_exclude, end_exclude)] # list of our excluded dates
        year_data = Ropt[name][str(year)] # subset the release data by year and apply the exclusion period to each
        for start_date, end_date in excluded_date_ranges:
            excluded_range = year_data[(year_data.index < start_date) | (year_data.index > end_date)]
            year_data = excluded_range
        remaining_dataframes.append(year_data) # put all the data we want in the same dataframe
        
    combined_dataframe = pd.concat(remaining_dataframes)
    
    max_release_freq[i,2] = ((combined_dataframe >= safe_release[i]-tolerance) & (combined_dataframe <= safe_release[i] + tolerance)).sum().sum()

#perfect
for i in range(len(names)):          
    print(names[i])
    name = names[i]
    remaining_dataframes = []
    for year in range(start_year, end_year+1):
        drawdown = params[name][1] #determine the dates we need excluded for 'name' reservoir
        start_exclude = pd.to_datetime(f"{year}-10-01")
        end_exclude = start_exclude + timedelta(days=drawdown)
        excluded_date_ranges = [(start_exclude, end_exclude)] # list of our excluded dates
        year_data = Rperf[name][str(year)] # subset the release data by year and apply the exclusion period to each
        for start_date, end_date in excluded_date_ranges:
            excluded_range = year_data[(year_data.index < start_date) | (year_data.index > end_date)]
            year_data = excluded_range
        remaining_dataframes.append(year_data) # put all the data we want in the same dataframe
        
    combined_dataframe = pd.concat(remaining_dataframes)
    
    max_release_freq[i,3] = ((combined_dataframe >= safe_release[i]-tolerance) & (combined_dataframe <= safe_release[i] + tolerance)).sum().sum()

#combine into one df
max_release_df = pd.DataFrame({
    'Baseline': max_release_freq[:,0],
    'Baseline-TOCS': max_release_freq[:,1],
    'Forecast': max_release_freq[:,2],
    'Perfect': max_release_freq[:,3]})
max_release_df.index = names
plt.rcParams['figure.figsize'] = [8, 6]

ax = max_release_df.plot(kind='bar', width=0.5, position=1, align='center', color = ['r','b','g','c'])
plt.xlabel('Reservoir')
plt.ylabel('Count')
plt.title('Frequency of maximum release by policy')


#%% Figure 8 - System dynamics
#df_hist = pd.read_csv('data/historical.csv', index_col=0, parse_dates=True)
df_hist = df_hist['2013-11-1':]

plt.rcParams['figure.figsize'] = [8, 8]

S_mean_base = np.zeros(len(names))
S_mean_opt = np.zeros(len(names))
S_delta = np.zeros(len(names))

for i in range(len(names)):
    name = names[i]
    S_mean_base[i] = Sbaseline[name].mean()/(capacity[i]*1000)
    S_mean_opt[i] = Sopt[name].mean()/(capacity[i]*1000)
    S_delta[i] = S_mean_opt[i] - S_mean_base[i]

#ratio of max inflow to reservoir storage
ratio_inflow_to_cap = np.zeros(len(names))

for i in range(len(names)):
    name = names[i]
    ratio_inflow_to_cap[i] = df_hist[name+'_inflow_cfs'].mean()*1.98347/(capacity[i]*params[name][4]*1000)


x = ratio_inflow_to_cap
y = S_delta
print(np.corrcoef(x,y))
fig, ax = plt.subplots(2,2)

ax[0,0].scatter(x,y)

for i, txt in enumerate(names):
    ax[0,0].annotate(txt, (x[i],y[i]))
ax[0,0].set(xlabel = 'Ratio of mean inflow to capacity (AF/AF)')
ax[0,0].set_title('(a)')
ax[0,0].set(ylabel = 'Normalized total surplus storage (AF/AF)')

#ratio of max inflow to max release
ratio_inflow_to_release = np.zeros(len(names))

for i in range(len(names)):
    name = names[i]
    #ratio_inflow_to_release[i] = (df_hist[name+'_inflow_cfs'].max()*1.98347)/(Ropt[name].max()*1.98347)
    ratio_inflow_to_release[i] = (df_hist[name+'_inflow_cfs'].max()*1.98347)/(safe_release[i]*1.98347)

x = ratio_inflow_to_release
y = S_delta
print(np.corrcoef(x,y))
ax[0,1].scatter(x,y)

for i, txt in enumerate(names):
    ax[0,1].annotate(txt, (x[i],y[i]))
ax[0,1].set(xlabel = 'Ratio of largest inflow to safe release (AF/AF)')
ax[0,1].set_title('(b)')
ax[0,1].set(ylabel = 'Normalized total surplus storage (AF/AF)')

ratio_safe_to_cap = np.zeros(len(names))

for i in range(len(names)):
    name = names[i]
    ratio_safe_to_cap[i] = safe_release[i]*1.98347/(capacity[i]*1000)

x = ratio_safe_to_cap
y = S_delta
ax[1,0].scatter(x,y)

for i, txt in enumerate(names):
    ax[1,0].annotate(txt, (x[i],y[i]))
ax[1,0].set(xlabel = 'Ratio of safe release to capacity (AF/AF)')
ax[1,0].set_title('(c)')
ax[1,0].set(ylabel = 'Normalized total surplus storage (AF/AF)')


x = nses[:,4]
y = S_delta
 
ax[1,1].scatter(x,y)
for i, txt in enumerate(names):
    ax[1,1].annotate(txt, (x[i],y[i]))
ax[1,1].set(xlabel = 'NSE at 4 days lead')
ax[1,1].set_title('(d)')
ax[1,1].set(ylabel = 'Normalized total surplus storage (AF/AF)')


plt.tight_layout()
plt.show()

#%%


# Ratio of inflow to capacity
x = ratio_inflow_to_cap
y = S_delta
print(np.corrcoef(x,y))

t_statistic, p_value = stats.ttest_ind(x, y)
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis, p-value: " + str(p_value))
else:
    print("Fail to reject the null hypothesis, p-value: " + str(p_value))

# Ratio of inflow to release
x = ratio_inflow_to_release
y = S_delta
print(np.corrcoef(x,y))

t_statistic, p_value = stats.ttest_ind(x, y)
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis, p-value: " + str(p_value))
else:
    print("Fail to reject the null hypothesis, p-value: " + str(p_value))    
# Ratio of safe release to capacity
x = ratio_safe_to_cap
y = S_delta
print(np.corrcoef(x,y))

t_statistic, p_value = stats.ttest_ind(x, y)
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis, p-value: " + str(p_value))
else:
    print("Fail to reject the null hypothesis, p-value: " + str(p_value))

# Forecast skill
x = nses[:,4]
y = S_delta
print(np.corrcoef(x,y))

t_statistic, p_value = stats.ttest_ind(x, y)
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis, p-value: " + str(p_value))
else:
    print("Fail to reject the null hypothesis, p-value: " + str(p_value))


#%% Flood benefits quantification
names = ['SHA','ORO','BUL','FOL','PAR','NHG','NML','DNP','EXC','MIL','PNF','TRM','SCC','ISB']

filtered_names = ['PAR','NML','DNP','EXC','MIL','PNF','TRM','SCC','ISB']

df_hist = df_hist['2013-11-1':]

plt.rcParams['figure.figsize'] = [8, 8]

flood_delta = (max_release_df['Baseline'] - max_release_df['Forecast'])/max_release_df['Baseline']

exclude_index = ['SHA','ORO','BUL','FOL','NHG']
mask = ~max_release_df.index.isin(exclude_index)
filtered_df = max_release_df[mask]

flood_delta = (filtered_df['Baseline'] - filtered_df['Forecast'])/filtered_df['Baseline']*100

#ratio of max inflow to reservoir storage
ratio_inflow_to_cap = np.zeros(len(flood_delta))
indices_to_ignore = [0,1,2,3,5] # create a mask to ignore these reservoirs
mask = np.ones(len(capacity), dtype=bool)
mask[indices_to_ignore] = False
filtered_capacity = capacity[mask]

for i in range(len(flood_delta)):
    name = filtered_names[i]
    ratio_inflow_to_cap[i] = df_hist[name+'_inflow_cfs'].mean()*1.98347/(filtered_capacity[i]*1000)


x = ratio_inflow_to_cap
y = flood_delta

print(np.corrcoef(x, y))
fig, ax = plt.subplots(2,2)

ax[0,0].scatter(x,y)

for i, txt in enumerate(filtered_names):
    ax[0,0].annotate(txt, (x[i],y[i]))
ax[0,0].set(xlabel = 'Ratio of mean inflow to capacity (AF/AF)')
ax[0,0].set_title('(a)')
ax[0,0].set(ylabel = '% reduction in frequency of max releases')

#ratio of max inflow to max release
ratio_inflow_to_release = np.zeros(len(filtered_names))

filtered_safe_release = safe_release[mask]

for i in range(len(flood_delta)):
    name = filtered_names[i]
    ratio_inflow_to_release[i] = (df_hist[name+'_inflow_cfs'].max()*1.98347)/(filtered_safe_release[i]*1.98347)

x = ratio_inflow_to_release
y = flood_delta

print(np.corrcoef(x, y))
ax[0,1].scatter(x,y)

for i, txt in enumerate(filtered_names):
    ax[0,1].annotate(txt, (x[i],y[i]))
ax[0,1].set(xlabel = 'Ratio of largest inflow to safe release (AF/AF)')
ax[0,1].set_title('(b)')
ax[0,1].set(ylabel = '% reduction in frequency of max releases')


ratio_safe_to_cap = np.zeros(len(flood_delta))

for i in range(len(flood_delta)):
    name = filtered_names[i]
    ratio_safe_to_cap[i] = filtered_safe_release[i]*1.98347/(filtered_capacity[i]*1000)

x = ratio_safe_to_cap
y = flood_delta

print(np.corrcoef(x, y))
ax[1,0].scatter(x,y)

for i, txt in enumerate(filtered_names):
    ax[1,0].annotate(txt, (x[i],y[i]))
ax[1,0].set(xlabel = 'Ratio of safe release to capacity (AF/AF)')
ax[1,0].set_title('(c)')
ax[1,0].set(ylabel = '% reduction in frequency of max releases')


x = nses[:,4][mask]
y = flood_delta

print(np.corrcoef(x, y))
 
ax[1,1].scatter(x,y)
for i, txt in enumerate(filtered_names):
    ax[1,1].annotate(txt, (x[i],y[i]))
ax[1,1].set(xlabel = 'NSE at 4 days lead')
ax[1,1].set_title('(d)')
ax[1,1].set(ylabel = '% reduction in frequency of max releases')


plt.tight_layout()
plt.show()
#%%

# Ratio of inflow to capacity
x = ratio_inflow_to_cap
y = flood_delta
print(np.corrcoef(x,y))

t_statistic, p_value = stats.ttest_ind(x, y)
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis, p-value: " + str(p_value))
else:
    print("Fail to reject the null hypothesis, p-value: " + str(p_value))

# Ratio of inflow to release
x = ratio_inflow_to_release
y = flood_delta
print(np.corrcoef(x,y))

t_statistic, p_value = stats.ttest_ind(x, y)
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis, p-value: " + str(p_value))
else:
    print("Fail to reject the null hypothesis, p-value: " + str(p_value))    
# Ratio of safe release to capacity
x = ratio_safe_to_cap
y = flood_delta
print(np.corrcoef(x,y))

t_statistic, p_value = stats.ttest_ind(x, y)
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis, p-value: " + str(p_value))
else:
    print("Fail to reject the null hypothesis, p-value: " + str(p_value))

# Forecast skill
x = nses[:,4][mask]
y = flood_delta
print(np.corrcoef(x,y))

t_statistic, p_value = stats.ttest_ind(x, y)
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis, p-value: " + str(p_value))
else:
    print("Fail to reject the null hypothesis, p-value: " + str(p_value))

#%% Figure 10 - Random Seed Results

seeds = pd.read_csv('data/seeds.csv', header = None)
plt.rcParams['figure.figsize'] = [8, 6]

plt.scatter(seeds.index, seeds.iloc[:,0])
plt.title('Objective Function Value by Random Seed')
plt.xlabel('Random Seed')
plt.ylabel('Objective Function Value')
plt.show()
print(seeds.std())

#%% Are baseline or observed policy releases going above the safe release value?
df_hist = pd.read_csv('data/historical.csv', index_col=0, parse_dates=True)
df_hist = df_hist['2013-11-1':]


for i in range(len(names)):
    safe = nodes[names[i]]['safe_release_cfs']
    name = names[i]
    plt.subplot(2,1,1)
    plt.plot(Rbaseline[name])
    plt.plot(Roptbase[name])
    plt.plot(Ropt[name])
    plt.plot(Rperf[name])
    plt.plot(df_hist[name+'_outflow_cfs'])
    plt.axhline(y = safe, color = 'r', linestyle = '-')
    plt.title(name)
    plt.ylabel('Release (cfs)')
    plt.legend(['Baseline','Baseline-TOCS','Forecast','Perfect','Observed'])
    plt.subplot(2,1,2)
    plt.plot(Sbaseline[name])
    plt.plot(Soptbase[name])
    plt.plot(Sopt[name])
    plt.plot(Sperf[name])
    plt.show()
    
#%% How much extra storage does FIRO provide?
# original: capacity * 1000 * flood pool TOCS value
# new: capacity * 1000 * flood pool TOCS value * P3 TOCS multiplier
# subtract to find difference
original = 0
new = 0
for i in range(len(names)):
    original += capacity[i] * 1000 * params[names[i]][4]
    new += capacity[i] * 1000 * params[names[i]][4] * opt_params[opt_params['Reservoir'] == names[i]]['P3'].values

print(new - original)

# print the original TOCS values
for i in range(len(names)):
    print(params[names[i]][4])




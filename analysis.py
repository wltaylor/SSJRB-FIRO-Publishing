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
import xarray as xr
import json
import matplotlib
from matplotlib import cm
import matplotlib.colors as colors
from matplotlib.cm import ScalarMappable
from util import *



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

start_date = datetime.datetime.strptime('11-01-2013', "%m-%d-%Y")
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
from mpl_toolkits.axes_grid1 import make_axes_locatable

fig = plt.figure(figsize = (16,6))

ax1 = fig.add_subplot(1,1,1, projection = crs.PlateCarree())

ax1.set_extent([-124, -117, 35, 42],
              crs=crs.PlateCarree()) ## Important

ax1.add_feature(cfeature.COASTLINE, rasterized=True)
ax1.add_feature(cfeature.LAND, color="cornsilk", alpha=1.0, rasterized=True)
ax1.add_feature(cfeature.BORDERS, linestyle="--", rasterized=True)
ax1.add_feature(cfeature.OCEAN, color="skyblue", alpha=0.4, rasterized=True)
ax1.add_feature(cfeature.RIVERS, edgecolor="skyblue", rasterized=True)
ax1.add_feature(cfeature.STATES, rasterized=True)
#plt.title("Sacramento, San Joaquin and Tulare Reservoirs")

for k in nodes:
    if nodes[k]['type'] == 'reservoir':
        lon, lat = nodes[k]['coords']

        if k not in ('CMN', 'SLF', 'LUS'):
            K = nodes[k]['capacity_taf']
            #plot reservoir markers
            ax1.scatter(lon, lat,
                       s = 3*np.sqrt(K), 
                       marker = '^', 
                       alpha = 0.5, 
                       zorder = 3, 
                       c = 'k', 
                       edgecolor = 'black', 
                       transform = crs.PlateCarree())
            #plot reservoir text name
            ax1.annotate(k, 
                        (lon+0.13, lat-0.01), 
                        transform = crs.PlateCarree(),
                        zorder=3)
#plot legend
for capacities in [500, 2500, 4500]:
    ax1.scatter([], [], c='k',
                alpha=0.5, 
                s=3*np.sqrt(capacities),
                label=capacities,
                marker = '^')
ax1.legend(scatterpoints=1, frameon=False, labelspacing=1, title='Reservoir Capacity', facecolor='white')

column_labels = ["Reservoir", "Capacity (TAF)"]
column_widths = [0.5,0.3]
cell_text = [
    ["Shasta (SHA)", "4579"], 
    ["Oroville (ORO)", "3524"], 
    ["New Bullards Bar (BUL)", "970"],
    ["Folsom (FOL)", "975"],
    ["Pardee (PAR)", "620"],
    ["New Hogan (NHG)", "317"],
    ["New Melones (NML)", "2369"],
    ["Don Pedro (DNP)", "2024"],
    ["Exchequer (EXC)", "1021"],
    ["Millerton (MIL)", "526"],
    ["Pine Flat (PNF)", "1000"],
    ["Terminus (TRM)", "185"],
    ["Lake Success (SCC)", "82"],
    ["Isabella (ISB)", "568"],
]




table = ax1.table(cellText=cell_text, 
                  colLabels=column_labels, 
                  colWidths=column_widths, 
                  cellLoc='center', 
                  loc='center right',
                  bbox=[1.0, 0, 0.5, 1])
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.2, 1.2)

plt.subplots_adjust(left=0.2, right=0.8)
# Hide the axis
#ax2.axis('off')

plt.savefig('Fig1.tif', format='tif', bbox_inches='tight', dpi=300)
plt.show()

#%% Figure 2 - Reservoir structure and risk curve on powerpoint
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Start by creating a figure and two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Left plot - Line plot
ax1.plot([0, 5, 14], [0, 0, 50], color='black')  # piecewise linear curve

ax1.annotate('', # left arrowhead
             xy=(4.5, 2), 
             xytext=(0.5, 2),
             arrowprops=dict(facecolor='black', width=1, headwidth=8),
             horizontalalignment='center', 
             color='black')
ax1.annotate('', # right arrowhead
             xy=(0.5, 2), 
             xytext=(4.5, 2),
             arrowprops=dict(facecolor='black', width=1, headwidth=8),
             horizontalalignment='center', 
             color='black')
ax1.annotate('X1', # text
             xy=(2.1,3),
             fontsize=16)

ax1.annotate('', 
             xy=(10, 30),
             xytext=(8, 30),
             arrowprops=dict(facecolor='black', width=1, headwidth=8),
             horizontalalignment='center', 
             color='black')
ax1.annotate('X2', # text
             xy=(6.8,29),
             fontsize=16)

ax1.set_xlabel('Forecast Lead Time (days)')
ax1.set_ylabel('Percent Risk (%)')
ax1.set_title('(a)')

# Right plot - Bar plot
# Add rectangles to represent the pools
flood_control = Rectangle((0.15, 0.8), 0.7, 0.2, facecolor='white', edgecolor='black')
conservation = Rectangle((0.15, 0.3), 0.7, 0.5, facecolor='white', edgecolor='black')
dead_pool = Rectangle((0.15, 0), 0.7, 0.3, facecolor='white',edgecolor='black')
ax2.add_patch(flood_control)
ax2.add_patch(conservation)
ax2.add_patch(dead_pool)

trapezoid_points = [(0.7, 0), (0.7, 1), (0.9, 1), (1, 0)]
trapezoid = matplotlib.patches.Polygon(trapezoid_points, closed=True, facecolor='grey', edgecolor='black')
ax2.add_patch(trapezoid)

# Add labels to the pools
ax2.text(0.45, 0.85, 'Flood control pool', horizontalalignment='center', verticalalignment='center', fontsize=12)
ax2.text(0.45, 0.55, 'Conservation pool', horizontalalignment='center', verticalalignment='center', fontsize=12)
ax2.text(0.45, 0.15, 'Dead pool', horizontalalignment='center', verticalalignment='center', fontsize=12)

# Draw arrow for X3
ax2.annotate('', # left arrowhead
             xy=(0.1, 0.65), 
             xytext=(0.1, 0.95),
             arrowprops=dict(facecolor='black', width=1, headwidth=8),
             horizontalalignment='center', 
             color='black')
ax2.annotate('', # right arrowhead
             xy=(0.1, 0.95), 
             xytext=(0.1, 0.65),
             arrowprops=dict(facecolor='black', width=1, headwidth=8),
             horizontalalignment='center', 
             color='black')

ax2.annotate('X3', xy=(0.02, 0.78), fontsize=16) # text
ax2.set_title('(b)')
# Adjust the layout
ax2.axis('off')
plt.tight_layout()
plt.savefig('Fig2.tif', format='tif', bbox_inches='tight', dpi=300)
plt.show()


#%% Figure 3 - Example forecast

from datetime import date
plt.rcParams['figure.figsize'] = [12, 6]
#nodes = json.load(open('data/nodes.json'))
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
plt.title('(a)')



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
    nses[site,lead] = nse(pred,obs)

    # false positive: prediction above 90th pctile, but observed is not
    FP = ((pred_max) > np.quantile(obs,0.9)) & (obs < np.quantile(obs,0.9))
    FPRs[site,lead] = FP.sum() / len(FP)
    # false negative: prediction below 90th pctile, but observed is above
    FN = ((pred_max) < np.quantile(obs,0.9)) & (obs > np.quantile(obs,0.9))
    FNRs[site,lead] = FN.sum() / len(FN)

names = ['SHA','ORO','BUL','FOL','PAR','NHG','NML','DNP','EXC','MIL','PNF','TRM','SCC','ISB']
xvals = [10, #SHA
         13, #ORO
         13, #BUL
         12, #FOL
         13, #PAR
         11, #NHG
         11, #NML
         13, #DNP
         12, #EXC
         13, #MIL
         12, #PNF
         12, #TRM
         12, #SCC
         12]#ISB

plt.subplot(1,2,2)
for k in range(len(names)):
    name = names[k]
    x = np.zeros(14)
    for i in range(len(x)):
        x[i] = nses[k,i]
    plt.plot(x, label=name)
    #plt.annotate(name, (10, x[10]))
    labelLines(plt.gca().get_lines(), xvals=xvals, zorder=2.5, color='black')

#plt.legend(names, loc='lower left')
plt.ylabel('NSE')
plt.xlabel('Lead time (days)')
plt.title('(b)')
plt.tight_layout()
plt.savefig('Fig3.tif', format='tif', bbox_inches='tight', dpi=300)
plt.show()


#%% Figure 4 - Average Storage per Policy and Max release frequency per policy
# Count the number of times the max release is hit for each policy
safe_release = np.zeros(len(names))
for i in range(len(names)):
    safe_release[i] = nodes[names[i]]['safe_release_cfs']

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

# Count the number of times the max release is hit for each policy, drawdown period excluded
max_release_freq = np.zeros((14,4))
start_year = 2013
end_year = 2023

params = json.load(open('data/params.json'))

# baseline
for i,name in enumerate(names):
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
for i,name in enumerate(names):          
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
for i,name in enumerate(names):          
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
for i,name in enumerate(names):          
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

capacity = np.zeros(len(names))
for i in range(len(capacity)):
    capacity[i] = nodes[names[i]]['capacity_taf']


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Plotting on the first subplot
for i in range(len(names)):
    res = names[i]
    ax1.scatter(res, Sbaseline.iloc[:,i].mean()/capacity[i]/1000, label='Baseline', color='r', alpha=0.5)
    ax1.scatter(res, Soptbase.iloc[:,i].mean()/capacity[i]/1000, label='Baseline-TOCS', color='b', alpha=0.5)
    ax1.scatter(res, Sopt.iloc[:,i].mean()/capacity[i]/1000, label='Forecast', color='g', alpha=0.5)
    ax1.scatter(res, Sperf.iloc[:,i].mean()/capacity[i]/1000, label='Perfect', color='c', alpha=0.5)

ax1.set_xlabel('Reservoir')
ax1.set_ylabel('Normalized Capacity (unitless)')
ax1.legend(['Baseline (J = 0.490)','Baseline-TOCS (J = 0.571)', 'Forecast (J = 0.585)', 'Perfect (J = 0.587)'])
ax1.set_title('(a)')

# Plotting on the second subplot
max_release_df.plot(kind='bar', width=0.5, position=1, align='center', color=['r','b','g','c'], ax=ax2, alpha = 0.5)
ax2.set_xlabel('Reservoir')
ax2.set_ylabel('Count')
ax2.set_title('(b)')

plt.tight_layout()
plt.savefig('Fig4.tif', format='tif', bbox_inches='tight', dpi=300)
plt.show()


#%% Figure 5 - Oroville storage and release plot

#show Oroville during 2017 spill
df_hist = pd.read_csv('data/historical.csv', index_col=0, parse_dates=True)
df_hist = df_hist['2013-11-1':]

from datetime import datetime
plt.rcParams['figure.figsize'] = [10, 5]
startdate = datetime(2017,1,15)
enddate = datetime(2017,2,28)
i = 1
plt.subplot(211)
plt.plot(Sbaseline.iloc[:,i],'r', alpha = 0.75, label='Baseline')
plt.plot(Soptbase.iloc[:,i],'b', alpha = 0.75, label='Baseline-TOCS')
plt.plot(Sopt.iloc[:,i],'g', alpha = 0.75, label='Forecast')
plt.plot(Sperf.iloc[:,i],'c', alpha = 0.75, label='Perfect')
plt.ylabel("Storage (AF)")
plt.title('(a)')
    
plt.subplot(212)
plt.plot(Rbaseline.iloc[:,i],'r', alpha = 0.75, label='Baseline')
plt.plot(Roptbase.iloc[:,i],'b', alpha = 0.75, label='Baseline-TOCS')
plt.plot(Ropt.iloc[:,i],'g', alpha = 0.75, label='Forecast')
plt.plot(Rperf.iloc[:,i],'c', alpha = 0.75, label='Perfect')
plt.plot(df_hist[names[i]+"_inflow_cfs"], 'black', alpha=0.75, label='Observed Inflow')
plt.xlabel("Date")
plt.ylabel("Release (cfs)")
plt.title('(b)')
plt.legend()
plt.xlim(startdate,enddate)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('Fig5.tif', format='tif', bbox_inches='tight', dpi=300)
plt.show()
#%% Figure 6 - Scatter Plot of Parameter Values, scalar and risk curve version
import matplotlib.gridspec as gridspec
from labellines import labelLines
names = ['SHA','ORO','BUL','FOL','PAR','NHG','NML','DNP','EXC','MIL','PNF','TRM','SCC','ISB']

# Load your data
baseopt_params = pd.read_csv('data/optimized_baseline_params.csv', header=None)
opt_params = pd.read_csv('data/optimized_params.csv')
opt_params = opt_params.drop(['Unnamed: 0'], axis=1)
perf_params = pd.read_csv('data/perfect_params.csv')
perf_params = perf_params.drop(['Unnamed: 0'], axis=1)
baseline_params = np.ones(14)

# Set up the colormap and normalization for the color variable
color_variable = opt_params.iloc[:,3]
cmap = plt.cm.get_cmap('viridis')  # Choose a colormap
norm = plt.Normalize(vmin=min(color_variable), vmax=max(color_variable))

# Create the figure and a GridSpec layout
plt.rcParams['figure.figsize'] = [14, 7]
fig = plt.figure(tight_layout=True)
gs1 = gridspec.GridSpec(1, 1, left = 0.05, right = 0.48)

ax1 = fig.add_subplot(gs1[0])
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

xvals = [12, #SHA
         10, #ORO
         8, #BUL
         12, #FOL
         11, #PAR
         12, #NHG
         8, #NML
         12, #DNP
         11, #EXC
         10, #MIL
         11, #PNF
         12, #TRM
         12, #SCC
         12]#ISB
# locations for labels

# Draw the second plot now on the shared axes
for k in range(14):  # Assuming 'names' has 14 items
    piecewise = np.zeros(14)
    for i in range(len(piecewise)):
        if i < opt_params.iloc[k,1]:
            piecewise[i] = 0
        else:
            piecewise[i] = (i - np.round(opt_params.iloc[k,1]))*opt_params.iloc[k,2]*100
    ax1.plot(piecewise, c=cmap(norm(color_variable[k])), label = names[k])
    labelLines(plt.gca().get_lines(), xvals=xvals, zorder=2.5, color='black')

cbar = fig.colorbar(sm, ax=ax1)
cbar.set_label('TOCS multiplier')
ax1.set_xlabel('Lead day')
ax1.set_ylabel('Risk percent (%)')
ax1.set_xlim(0,13)
ax1.set_title('(a)')

gs2 = gridspec.GridSpec(3,1, left = 0.56, right = 0.98, hspace = 0.45)

# First subplot
ax0 = fig.add_subplot(gs2[0, 0])
ax0.scatter(opt_params.iloc[:,0], opt_params.iloc[:,1], color = 'g')
ax0.set_title('(b)')
ax0.set_ylim(0,10)
ax0.set_ylabel('Days With No Risk')

# Second subplot
ax1 = fig.add_subplot(gs2[1, 0])
ax1.scatter(opt_params.iloc[:,0], opt_params.iloc[:,2], color = 'g')
ax1.set_ylim(0,0.1)
ax1.set_ylabel('Slope of Risk Curve')
ax1.set_title('(c)')

# Third subplot
ax2 = fig.add_subplot(gs2[2, 0])
ax2.scatter(opt_params.iloc[:,0], baseline_params, color = 'r')
ax2.scatter(opt_params.iloc[:,0], baseopt_params.iloc[:,0], color = 'b')
ax2.scatter(opt_params.iloc[:,0], opt_params.iloc[:,3], color = 'g')
ax2.scatter(perf_params.iloc[:,0], perf_params.iloc[:,3], color = 'c')
ax2.legend(['Baseline','Baseline-TOCS','Forecast','Perfect'], loc='lower right')
ax2.set_ylim(1.0,2.0)
ax2.set_ylabel('TOCS Multiplier')
ax2.set_title('(d)')

plt.savefig('Fig6.tif', format='tif', bbox_inches='tight', dpi=300)
plt.show()

#%% Figure 7 - Supply factors analysis
from scipy.stats import pearsonr, linregress
params = json.load(open('data/params.json'))

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
fig, ax = plt.subplots(2,2)

ax[0,0].scatter(x,y)

for i, txt in enumerate(names):
    ax[0,0].annotate(txt, (x[i],y[i]))
ax[0,0].set(xlabel = 'Ratio of mean inflow to capacity (AF/AF)')
ax[0,0].set_title('(a)')
ax[0,0].set(ylabel = 'Normalized total surplus storage (AF/AF)')

# calculate corr coef and p value
corr_coef, p_val = pearsonr(x,y)
box_props = dict(boxstyle='round', facecolor='white', alpha=0.5)

ax[0,0].annotate(f'Corr: {corr_coef:.2f}\nP-value: {p_val:.2f}', xy = (0.65,0.85), xycoords='axes fraction', bbox = box_props)
# Add line of best fit
slope, intercept, _, _, _ = linregress(x, y)
ax[0, 0].plot(x, slope * x + intercept, color='red', linestyle='-', label='Line of Best Fit')


#ratio of max inflow to max release
ratio_inflow_to_release = np.zeros(len(names))

for i in range(len(names)):
    name = names[i]
    ratio_inflow_to_release[i] = (df_hist[name+'_inflow_cfs'].max()*1.98347)/(safe_release[i]*1.98347)

x = ratio_inflow_to_release
y = S_delta
ax[0,1].scatter(x,y)

for i, txt in enumerate(names):
    ax[0,1].annotate(txt, (x[i],y[i]))
ax[0,1].set(xlabel = 'Ratio of largest inflow to safe release (AF/AF)')
ax[0,1].set_title('(b)')
ax[0,1].set(ylabel = 'Normalized total surplus storage (AF/AF)')

# calculate corr coef and p value
corr_coef, p_val = pearsonr(x,y)
ax[0,1].annotate(f'Corr: {corr_coef:.2f}\nP-value: {p_val:.2f}', xy = (0.05,0.85), xycoords='axes fraction', bbox=box_props)

# Add line of best fit
slope, intercept, _, _, _ = linregress(x, y)
ax[0,1].plot(x, slope * x + intercept, color='red', linestyle='-', label='Line of Best Fit')


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

# calculate corr coef and p value
corr_coef, p_val = pearsonr(x,y)
ax[1,0].annotate(f'Corr: {corr_coef:.2f}\nP-value: {p_val:.2f}', xy = (0.65,0.85), xycoords='axes fraction', bbox=box_props)

# Add line of best fit
slope, intercept, _, _, _ = linregress(x, y)
ax[1, 0].plot(x, slope * x + intercept, color='red', linestyle='-', label='Line of Best Fit')


x = nses[:,13]
y = S_delta
 
ax[1,1].scatter(x,y)
for i, txt in enumerate(names):
    ax[1,1].annotate(txt, (x[i],y[i]))
ax[1,1].set(xlabel = 'NSE at 4 days lead')
ax[1,1].set_title('(d)')
ax[1,1].set(ylabel = 'Normalized total surplus storage (AF/AF)')

# calculate corr coef and p value
corr_coef, p_val = pearsonr(x,y)
ax[1,1].annotate(f'Corr: {corr_coef:.2f}\nP-value: {p_val:.2f}', xy = (0.05,0.85), xycoords='axes fraction', bbox=box_props)

# Add line of best fit
slope, intercept, _, _, _ = linregress(x, y)
ax[1,1].plot(x, slope * x + intercept, color='red', linestyle='-', label='Line of Best Fit')

plt.tight_layout()
plt.savefig('Fig7.tif', format='tif', bbox_inches='tight', dpi=300)
plt.show()
#%% Supply factors OLS (supplemental material)
import statsmodels.api as sm

supply_factors = pd.DataFrame({'mean_inflow_to_capacity': ratio_inflow_to_cap,
                               'largest_inflow_to_safe_release': ratio_inflow_to_release,
                               'ratio_safe_release_to_capacity': ratio_safe_to_cap,
                               'NSE': nses[:,4]})

X = supply_factors
X = sm.add_constant(X)
y = S_delta
model = sm.OLS(y,X)
results = model.fit()
results.summary()

coefficients = results.params
t_values = results.tvalues
p_values = results.pvalues
adj_r_squared = results.rsquared_adj
f_statistic = results.fvalue
# Combine the coefficients, standard errors, t-values, and p-values into a DataFrame
summary_df = pd.DataFrame({'Coefficient': coefficients,
                            't-value': t_values,
                            'p-value': p_values,
                            'adjusted R squared': adj_r_squared,
                            'F statistic': f_statistic})

# Print the summary DataFrame
print(summary_df)

r_squared = results.rsquared
adj_r_squared = results.rsquared_adj
f_statistic = results.fvalue
coefficients = results.params
p_values = results.pvalues

# Combine the statistics into a DataFrame
summary_df = pd.DataFrame({'Coefficient': coefficients,
                            'Coefficient Probability': p_values})

# Add R-squared, adjusted R-squared, and F-statistic to the summary DataFrame
summary_df.loc['R-squared'] = r_squared
summary_df.loc['Adjusted R-squared'] = adj_r_squared
summary_df.loc['F-statistic'] = f_statistic

latex_table = summary_df.to_latex(float_format="%.4f", escape=False)

# Print the LaTeX table
print(latex_table)

#%% Figure 8 flood benefits
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

# calculate corr coef and p value
corr_coef, p_val = pearsonr(x,y)
ax[0,0].annotate(f'Corr: {corr_coef:.2f}\nP-value: {p_val:.2f}', xy = (0.65,0.85), xycoords='axes fraction', bbox=box_props)

# Add line of best fit
slope, intercept, _, _, _ = linregress(x, y)
ax[0,0].plot(x, slope * x + intercept, color='red', linestyle='-', label='Line of Best Fit')


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

# calculate corr coef and p value
corr_coef, p_val = pearsonr(x,y)
ax[0,1].annotate(f'Corr: {corr_coef:.2f}\nP-value: {p_val:.2f}', xy = (0.05,0.85), xycoords='axes fraction', bbox=box_props)

# Add line of best fit
slope, intercept, _, _, _ = linregress(x, y)
ax[0,1].plot(x, slope * x + intercept, color='red', linestyle='-', label='Line of Best Fit')


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

# calculate corr coef and p value
corr_coef, p_val = pearsonr(x,y)
ax[1,0].annotate(f'Corr: {corr_coef:.2f}\nP-value: {p_val:.2f}', xy = (0.65,0.85), xycoords='axes fraction', bbox=box_props)

# Add line of best fit
slope, intercept, _, _, _ = linregress(x, y)
ax[1,0].plot(x, slope * x + intercept, color='red', linestyle='-', label='Line of Best Fit')


x = nses[:,4][mask]
y = flood_delta

print(np.corrcoef(x, y))
 
ax[1,1].scatter(x,y)
for i, txt in enumerate(filtered_names):
    ax[1,1].annotate(txt, (x[i],y[i]))
ax[1,1].set(xlabel = 'NSE at 4 days lead')
ax[1,1].set_title('(d)')
ax[1,1].set(ylabel = '% reduction in frequency of max releases')

# calculate corr coef and p value
corr_coef, p_val = pearsonr(x,y)
ax[1,1].annotate(f'Corr: {corr_coef:.2f}\nP-value: {p_val:.2f}', xy = (0.05,0.85), xycoords='axes fraction', bbox=box_props)

# Add line of best fit
slope, intercept, _, _, _ = linregress(x, y)
ax[1,1].plot(x, slope * x + intercept, color='red', linestyle='-', label='Line of Best Fit')

plt.tight_layout()
plt.savefig('Fig8.tif', format='tif', bbox_inches='tight', dpi=300)
plt.show()

#%% Flood factors OLS (supplemental file)
# place factors into a df
flood_factors = pd.DataFrame({'mean_inflow_to_capacity': ratio_inflow_to_cap,
                               'largest_inflow_to_safe_release': ratio_inflow_to_release,
                               'ratio_safe_release_to_capacity': ratio_safe_to_cap,
                               'NSE': nses[:,4][mask]})

X = flood_factors
X = sm.add_constant(X)
y = flood_delta.values
model = sm.OLS(y,X)
results = model.fit()
results.summary()

#%% Figure 9 
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
import model
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

plt.rcParams['figure.figsize'] = [12, 6]

fig = plt.figure()

# define first gridspace and axes (left)
# simulate model for each seed and calculate J-value
gs1 = GridSpec(1, 1, left = 0.05, right = 0.48)
ax1 = fig.add_subplot(gs1[0])

for j in range(10):
    file_name = f"data/params_{j}.csv" # read in the 10 different seed results
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
        if j == 4:
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

plt.title('(a)')

# define second gridspace and axes (right)
# read in parameter values and plot
gs2 = GridSpec(3, 1, left = 0.54, right = 0.98, hspace = 0.45)
opt_params = pd.read_csv('data/optimized_params.csv')
opt_params = opt_params.drop(['Unnamed: 0'], axis=1)
params0 = pd.read_csv('data/params_0.csv', header=None)
params1 = pd.read_csv('data/params_1.csv', header=None)
params2 = pd.read_csv('data/params_2.csv', header=None)
params3 = pd.read_csv('data/params_3.csv', header=None)
params4 = pd.read_csv('data/params_4.csv', header=None)
params5 = pd.read_csv('data/params_5.csv', header=None)
params6 = pd.read_csv('data/params_6.csv', header=None)
params7 = pd.read_csv('data/params_7.csv', header=None)
params8 = pd.read_csv('data/params_8.csv', header=None)
params9 = pd.read_csv('data/params_9.csv', header=None)

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
plt.title('(b)')

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
plt.title('(c)')

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
plt.title('(d)')

plt.savefig('Fig9.tif', format='tif', bbox_inches='tight', dpi=300)
plt.show()

#%% Figure 10 LOOCV simulation

import matplotlib.dates as mdates
filename = f"data/params_loocv_{2017}.json"
params = json.load(open(filename))
params_in = tuple(np.array(v) for k,v in params.items())
results = model.simulate(params_in, Kr, Kp, *input_data, max_release, ramping_rate, use_firo=True, Qf=Qf)
df_opt = results_to_df(results, index=df_hist.index, res_keys=rk)
df_hist['water_year'] = df_hist.index.year
df_hist.loc[(df_hist.index.month >= 10) & (df_hist.index.month <= 12), 'water_year'] += 1
water_years = df_hist['water_year'].values
unique_water_years = np.unique(water_years)

mask = (water_years == 2017)
k = names[4] #PAR

plt.figure(figsize=(10,6))

ax1 = plt.subplot(211)
df_opt[k+'_storage_af'][mask].plot(color='red', label='FIRO Policy')
TOCS = df_opt[k+'_tocs_fraction'] * nodes[k]['capacity_taf'] * 1000
TOCS.plot(color='red', linestyle='--', label='FIRO Pool Limit')
plt.legend(loc='upper left')
plt.ylabel('Storage (AF)')

ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
plt.setp(ax1.get_xticklabels(), visible=False)  # Hide x-tick labels for the top subplot
plt.title('(a)')

ax2 = plt.subplot(212, sharex=ax1)
df_opt[k+'_outflow_cfs'][mask].plot(color='red', label = 'FIRO Policy')
df_hist[k+'_inflow_cfs'][mask].plot(color='grey', alpha = 0.5, label = 'Observed Inflow')

plt.axhline(max_release[4], alpha = 0.5, label = 'Max Safe Release Value')
plt.legend(loc='upper left')
plt.ylabel('Release (cfs)')
plt.xlabel('Water Year 2017')
plt.gcf().autofmt_xdate()

plt.title('(b)')
plt.tight_layout()
plt.subplots_adjust(hspace=0.2)

plt.savefig('Fig10.tif', format='tif', bbox_inches='tight', dpi=300)
plt.show()
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import json
import model
from scipy.optimize import differential_evolution as DE
import xarray as xr
from util import *
import os

sd = '2013-11-01'
nodes = json.load(open('data/nodes.json'))
df_hist = pd.read_csv('data/historical.csv', index_col=0, parse_dates=True)
medians = pd.read_csv('data/historical_medians.csv', index_col=0)

names = ['SHA','ORO','BUL','FOL','PAR','NHG','NML','DNP','EXC','MIL','PNF','TRM','SCC','ISB']

# the non-firo params have already been calibrated in train_historical.py
params = json.load(open('data/params.json'))
df_hist = pd.read_csv('data/historical.csv', index_col=0, parse_dates=True)[sd:]
rk = [k for k in nodes.keys() if (nodes[k]['type'] == 'reservoir') and (nodes[k]['fit_policy'])]
Kr = np.array([nodes[k]['capacity_taf'] * 1000 for k in rk])
max_release = np.array([nodes[k]['safe_release_cfs'] for k in rk])
ramping_rate = np.array([nodes[k]['ramping_rate'] for k in rk])
pk = [k for k in nodes.keys() if (nodes[k]['type'] == 'pump') and (nodes[k]['fit_policy'])]
Kp = np.array([nodes[k]['capacity_cfs'] for k in pk])

for i,r in enumerate(rk):
  params[r] += [0,0,0] # append three empty values for the firo policy

# forecast data
Qf = xr.open_dataset('data/cnrfc14d.nc')['Qf']
Qf = Qf.values[:,:,:40,:]
Qf.sort(axis=2)


# run simulation and put results in dataframe
input_data = get_simulation_data(rk, pk, df_hist, medians, init_storage=True)

def opt_wrapper(x, mask):

  for i,r in enumerate(rk):
    params[r][-3:] = x[i*3:(i+1)*3] # assign parameters
  params_in = tuple(np.array(v) for k,v in params.items())
  R,S,Delta,tocs = model.simulate(params_in, Kr, Kp, *input_data, max_release, ramping_rate, use_firo=True, Qf=Qf)
  
  # apply the mask
  R_masked = R[mask]
  S_masked = S[mask]
  Delta_masked = Delta[mask]
  tocs_masked = tocs[mask]
  # evaluate obj function
  obj = model.objective(R_masked,S_masked,Delta_masked,Kr, max_release)

  print(obj)
  return obj

bounds = [(0,10), (0,0.10), (0.8,2.0)]*14

# add a water year column to df_hist
df_hist['water_year'] = df_hist.index.year
df_hist.loc[(df_hist.index.month >= 10) & (df_hist.index.month <= 12), 'water_year'] += 1
water_years = df_hist['water_year'].values
unique_water_years = np.unique(water_years)
#%%

for i, water_year in enumerate(unique_water_years):
    print(water_year)
    mask = (water_years != water_year)
     
    opt = DE(opt_wrapper, bounds=bounds, args=(mask,), disp=True, maxiter=200, seed=0, polish=True)
    print(opt)
    
    # save to a file labeled by the held out year
    filename = f"data/params_loocv_{water_year}.json"  # Using f-string formatting

    with open(filename, 'w') as f:
        json.dump(params, f, indent=2)


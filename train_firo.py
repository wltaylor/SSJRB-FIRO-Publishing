import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import json
import model
from scipy.optimize import differential_evolution as DE, minimize
import xarray as xr
from util import *
import os

sd = '2013-11-01'
nodes = json.load(open('data/nodes.json'))
df_hist = pd.read_csv('data/historical.csv', index_col=0, parse_dates=True)[sd:]
medians = pd.read_csv('data/historical_medians.csv', index_col=0)

# the non-firo params have already been calibrated in train_historical.py
params = json.load(open('data/params.json'))

rk = [k for k in nodes.keys() if (nodes[k]['type'] == 'reservoir') and (nodes[k]['fit_policy'])]
Kr = np.array([nodes[k]['capacity_taf'] * 1000 for k in rk])
max_release = np.array([nodes[k]['safe_release_cfs'] for k in rk])
ramping_rate = np.array([nodes[k]['ramping_rate'] for k in rk])
pk = [k for k in nodes.keys() if (nodes[k]['type'] == 'pump') and (nodes[k]['fit_policy'])]
Kp = np.array([nodes[k]['capacity_cfs'] for k in pk])

for i,r in enumerate(rk):
  params[r] += [0,0,0] # append three empty values for the firo policy

# forecast data
forecast = 'HEFS'
if forecast == 'HEFS':
    Qf = xr.open_dataset('data/cnrfc14d.nc')['Qf']
    # <xarray.DataArray 'Qf' (site: 10, time: 3212, ens: 60, lead: 14)>
    Qf = Qf.values[:,:,:40,:] # numpy array, keep 40 ens members as limiting value
    Qf.sort(axis=2) # should help with runtime later - pre-sort the ensemble
# otherwise use the perfect forecast
else:
    Qf = xr.open_dataset('data/perfect_forecast.nc')['inflow']
    Qf = Qf.values

# run simulation and put results in dataframe
input_data = get_simulation_data(rk, pk, df_hist, medians, init_storage=True)

# input x - 3 firo policy parameters for each reservoir, 42 total
# this optimizes all parameters at once using average objective across all reservoirs
# it could also be done separately for each reservoir

def opt_wrapper(x):

  for i,r in enumerate(rk):
    params[r][-3:] = x[i*3:(i+1)*3] # assign parameters
  params_in = tuple(np.array(v) for k,v in params.items())
  R,S,Delta,tocs = model.simulate(params_in, Kr, Kp, *input_data, max_release, ramping_rate, use_firo=True, Qf=Qf)
  obj = model.objective(R,S,Delta,Kr, max_release)

  print(obj)
  return obj

# input x - 3 firo policy parameters for each reservoir, 42 total
# parameter 1: # days to allow zero risk (0-10)
# parameter 2: slope of risk/day allowed after that (0-0.10)
# parameter 3: multiplier to change tocs (0.8-2.0)
#bounds = [(0,10), (0,0.10), (0.8,2.0)]*14
#seed = np.arange(0,10,1)
#results = np.zeros(len(seed))

#for i in range(len(seed)):
#  opt = DE(opt_wrapper, bounds = bounds, disp=True, maxiter=200, seed=i, polish = True)
#  print(opt)
#  results[i] = opt.fun
#  param_results = np.zeros((14,3))
#  x = opt.x.tolist()
#  for j in range(0,14):
#    param_results[j,0] = x[j*3]
#    param_results[j,1] = x[(j*3)+1]
#    param_results[j,2] = x[(j*3)+2]
#  filename = f"params_{i}.csv"
#  np.savetxt(filename, param_results, delimiter=',')

#np.savetxt('data/seeds.csv', results, delimiter=",")

opt = DE(opt_wrapper, bounds = bounds, disp=True, maxiter=200, seed=4, polish = True)
print(opt)

# save the optimized parameters
x = opt.x.tolist()
for i,r in enumerate(rk):
  params[r][-3:] = x[i*3:(i+1)*3] # assign parameters

if forecast == 'HEFS':
    with open('data/params_firo.json', 'w') as f:
        json.dump(params, f, indent=2)
else:
    with open('data/perfect_params_firo.json', 'w') as f:
        json.dump(params, f, indent=2)


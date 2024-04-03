import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import json
import model
import copy
from util import *

sd = '2013-11-01'
nodes = json.load(open('data/nodes.json'))
df_hist = pd.read_csv('data/historical.csv', index_col=0, parse_dates=True)[sd:]
medians = pd.read_csv('data/historical_medians.csv', index_col=0)

params = json.load(open('data/params.json')) # baseline parameters

rk = [k for k in nodes.keys() if (nodes[k]['type'] == 'reservoir') and (nodes[k]['fit_policy'])]
Kr = np.array([nodes[k]['capacity_taf'] * 1000 for k in rk])
max_release = np.array([nodes[k]['safe_release_cfs'] for k in rk])
ramping_rate = np.array([nodes[k]['ramping_rate'] for k in rk])
pk = [k for k in nodes.keys() if (nodes[k]['type'] == 'pump') and (nodes[k]['fit_policy'])]
Kp = np.array([nodes[k]['capacity_cfs'] for k in pk])
input_data = get_simulation_data(rk, pk, df_hist, medians, init_storage=True)

# for each reservoir
# increase TOCS multiplier as high as possible without exceeding safe release
tocs_multipliers = np.zeros(len(rk))
params_temp = copy.deepcopy(params)

for i,k in enumerate(rk):
  print(k)
  original_tocs = params[k][4] 

  for tm in np.arange(1.0,2,0.001): # lower bound down to 0.5 because of PNF    
    params_temp[k][4] = original_tocs * tm
    params_in = tuple(np.array(v) for k,v in params_temp.items()) 

    R,S,Delta,tocs = model.simulate(params_in, Kr, Kp, *input_data, max_release, ramping_rate)
  
    temp = R[:,i]
    temp = pd.DataFrame(temp)
    temp = temp.diff()
    temp = temp.abs()
    max_rate = temp.max()
    max_rate = max_rate.values
    
    if max_rate > ramping_rate[i]+1 or R[:,i].max() > max_release[i]+1:
      print(max_rate)
      print(ramping_rate[i])
      print("max release is:"+str(max_release[i]))
      print(R[:,i].max())
      params_temp[k][4] = original_tocs * (tm - 0.001) # revert to previous value, did not exceed

      break      

    

  print('TOCS Multiplier = %0.4f' % (tm - 0.001))
  tocs_multipliers[i] = (tm - 0.001)

# once all multipliers found, run one more time to find the objective function
# note this does not change the original params.json values
params_in = tuple(np.array(v) for k,v in params_temp.items())
R,S,Delta,tocs = model.simulate(params_in, Kr, Kp, *input_data, max_release, ramping_rate)
obj = model.objective(R,S,Delta,Kr, max_release)
print('Total Objective Function = ', obj)

# save storage and release values for outside analysis
np.savetxt("data/optimized_baseline_release.csv", R, delimiter=",")
np.savetxt("data/optimized_baseline_storage.csv", S, delimiter=",")
np.savetxt("data/optimized_baseline_params.csv", tocs_multipliers, delimiter=",")



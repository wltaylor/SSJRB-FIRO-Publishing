import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import json
import model
import xarray as xr
import datetime
from datetime import datetime, timedelta
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

# baseline simulation
input_data = get_simulation_data(rk, pk, df_hist, medians, init_storage=True)
params_in = tuple(np.array(v) for k,v in params.items())
results = model.simulate(params_in, Kr, Kp, *input_data, max_release, ramping_rate)
df_baseline = results_to_df(results, index=df_hist.index, res_keys=rk)
df_baseline.to_csv('output/sim_baseline.csv') # save csv output for baseline run

# forecast data

forecast = 'HEFS'
if forecast == 'HEFS':
    Qf = xr.open_dataset('data/cnrfc14d.nc')['Qf']
    # <xarray.DataArray 'Qf' (site, time, ens, lead)>
    Qf = Qf.values[:,:,:40,:] # numpy array, keep 40 ens members as limiting value
    Qf.sort(axis=2) # should help with runtime later - pre-sort the ensemble
    params = json.load(open('data/params_firo.json'))

else:
    Qf = xr.open_dataset('data/perfect_forecast.nc')['inflow']
    Qf = Qf.values
    params = json.load(open('data/perfect_params_firo.json'))

#%%
params_in = tuple(np.array(v) for k,v in params.items())
results = model.simulate(params_in, Kr, Kp, *input_data, max_release, ramping_rate, use_firo=True, Qf=Qf)
df_opt = results_to_df(results, index=df_hist.index, res_keys=rk)

Rbase_opt = pd.read_csv('data/optimized_baseline_release.csv', header = None) 
Sbase_opt = pd.read_csv('data/optimized_baseline_storage.csv', header = None) 
start_date = datetime.strptime('11-01-2013', "%m-%d-%Y")
names = ['SHA','ORO','BUL','FOL','PAR','NHG','NML','DNP','EXC','MIL','PNF','TRM','SCC','ISB']

K = len(Rbase_opt)
dates = pd.date_range(start_date, periods=K)
Rbase_opt = pd.DataFrame(data = Rbase_opt.values, index = dates, columns = names)
Sbase_opt = pd.DataFrame(data = Sbase_opt.values, index = dates, columns = names)
#%% plotting

for i in range(len(names)):
    k = names[i]
    plt.subplot(311)
    
    df_baseline[k+'_storage_af'].plot(color='blue')
    TOCS = df_baseline[k+'_tocs_fraction'] * nodes[k]['capacity_taf'] * 1000
    TOCS.plot(color='blue', linestyle='--')
    Sbase_opt[k].plot(color='green')
    df_opt[k+'_storage_af'].plot(color='red')
    TOCS = df_opt[k+'_tocs_fraction'] * nodes[k]['capacity_taf'] * 1000
    TOCS.plot(color='red', linestyle='--')
    df_hist[k+'_storage_af'].plot(color='0.5')
    
    plt.legend(['Baseline', 'TOCS', 'Optimized_baseline','Optimized', 'TOCS', 'Obs'])
    plt.ylabel('Storage (AF)')
    
    plt.subplot(312)
    
    df_baseline[k+'_outflow_cfs'].plot(color='blue')
    Rbase_opt[k].plot(color='green')
    df_opt[k+'_outflow_cfs'].plot(color='red')
    df_hist[k+'_outflow_cfs'].plot(color='0.5')
    
    plt.legend(['Baseline', 'Optimized', 'Obs'])
    plt.ylabel('Release (cfs)')
    plt.suptitle(names[i])
    plt.show()

#%% export data
Rbaseline = df_baseline.loc[:,['SHA_outflow_cfs','ORO_outflow_cfs','BUL_outflow_cfs','FOL_outflow_cfs','PAR_outflow_cfs','NHG_outflow_cfs','NML_outflow_cfs','DNP_outflow_cfs','EXC_outflow_cfs','MIL_outflow_cfs', 'PNF_outflow_cfs', 'TRM_outflow_cfs', 'SCC_outflow_cfs', 'ISB_outflow_cfs']]
Ropt = df_opt.loc[:,['SHA_outflow_cfs','ORO_outflow_cfs','BUL_outflow_cfs','FOL_outflow_cfs','PAR_outflow_cfs','NHG_outflow_cfs','NML_outflow_cfs','DNP_outflow_cfs','EXC_outflow_cfs','MIL_outflow_cfs', 'PNF_outflow_cfs', 'TRM_outflow_cfs', 'SCC_outflow_cfs', 'ISB_outflow_cfs']]

Sbaseline = df_baseline.loc[:,['SHA_storage_af','ORO_storage_af','BUL_storage_af','FOL_storage_af','PAR_storage_af','NHG_storage_af','NML_storage_af','DNP_storage_af','EXC_storage_af','MIL_storage_af', 'PNF_storage_af', 'TRM_storage_af', 'SCC_storage_af', 'ISB_storage_af']]
Sopt = df_opt.loc[:,['SHA_storage_af','ORO_storage_af','BUL_storage_af','FOL_storage_af','PAR_storage_af','NHG_storage_af','NML_storage_af','DNP_storage_af','EXC_storage_af','MIL_storage_af', 'PNF_storage_af', 'TRM_storage_af', 'SCC_storage_af', 'ISB_storage_af']]

#save baseline results
np.savetxt("data/baseline_release.csv", Rbaseline, delimiter=",")
np.savetxt("data/baseline_storage.csv", Sbaseline, delimiter=",")

if forecast == 'HEFS':
    #save optimized results (with forecast)
    np.savetxt("data/optimized_storage.csv", Sopt, delimiter=",")
    np.savetxt("data/optimized_release.csv", Ropt, delimiter=",")
else:
    #save perfect results
    np.savetxt("data/perfect_release.csv", Ropt, delimiter=",")
    np.savetxt("data/perfect_storage.csv", Sopt, delimiter=",")

#optimized-baseline results are generated via simulate_baseline_tocs.py

# save parameters for plotting comparison
names = ['SHA','ORO','BUL','FOL','PAR','NHG','NML','DNP','EXC','MIL','PNF','TRM','SCC','ISB']

x = np.zeros((14,3))

for k in range(0,14):
    x[k,0] = params_in[k][-3] # parameter 1: # days to allow zero risk
    x[k,1] = params_in[k][-2] # parameter 2: slope of risk/day allowed after P1
    x[k,2] = params_in[k][-1] # parameter 3: tocs multiplier

params_df = pd.DataFrame(data = x,
                          columns = ['P1','P2','P3'],
                          index = names)
params_df = params_df.reset_index()
params_df = params_df.rename(columns={"index":"Reservoir"})

if forecast == 'HEFS':
    params_df.to_csv("data/optimized_params.csv")
else:
    params_df.to_csv("data/perfect_params.csv")

Delta = np.ones(14)
max_release = max_release.astype(np.float64)

res = model.objective(Rbaseline.values, Sbaseline.values, Delta, Kr, max_release)
print('The baseline (no forecast) objective function value is ' +str(res))

res = model.objective(Ropt.values, Sopt.values, Delta, Kr, max_release)
print('The optimized objective function value is ' +str(res))





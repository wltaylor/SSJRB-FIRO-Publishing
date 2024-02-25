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

# forecast data

forecast = 'CNRFC'
if forecast == 'CNRFC':
    Qf = xr.open_dataset('data/cnrfc14d.nc')['Qf']
    # <xarray.DataArray 'Qf' (site, time, ens, lead)>
    Qf = Qf.values[:,:,:40,:] # numpy array, keep 40 ens members as limiting value
    Qf.sort(axis=2) # should help with runtime later - pre-sort the ensemble
    params = json.load(open('data/params_firo_loocv.json'))

else:
    Qf = xr.open_dataset('data/perfect_forecast.nc')['inflow']
    Qf = Qf.values
    params = json.load(open('data/perfect_params_firo.json'))


params_in = tuple(np.array(v) for k,v in params.items())
print(params_in)
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
    
    # plt.legend(['Baseline', 'Highest Risk', 'Lowest Risk', 'Optimized'])
    plt.legend(['Baseline', 'TOCS', 'Optimized_baseline','Optimized', 'TOCS', 'Obs'])
    plt.ylabel('Storage (AF)')
    
    plt.subplot(312)
    
    df_baseline[k+'_outflow_cfs'].plot(color='blue')
    Rbase_opt[k].plot(color='green')
    df_opt[k+'_outflow_cfs'].plot(color='red')
    df_hist[k+'_outflow_cfs'].plot(color='0.5')
    
    # plt.legend(['Baseline', 'Highest Risk', 'Lowest Risk', 'Optimized'])
    plt.legend(['Baseline', 'Optimized', 'Obs'])
    plt.ylabel('Release (cfs)')
    plt.suptitle(names[i])
    plt.show()


plt.show()


#%% export data
Rbaseline = df_baseline.loc[:,['SHA_outflow_cfs','ORO_outflow_cfs','BUL_outflow_cfs','FOL_outflow_cfs','PAR_outflow_cfs','NHG_outflow_cfs','NML_outflow_cfs','DNP_outflow_cfs','EXC_outflow_cfs','MIL_outflow_cfs', 'PNF_outflow_cfs', 'TRM_outflow_cfs', 'SCC_outflow_cfs', 'ISB_outflow_cfs']]
Ropt = df_opt.loc[:,['SHA_outflow_cfs','ORO_outflow_cfs','BUL_outflow_cfs','FOL_outflow_cfs','PAR_outflow_cfs','NHG_outflow_cfs','NML_outflow_cfs','DNP_outflow_cfs','EXC_outflow_cfs','MIL_outflow_cfs', 'PNF_outflow_cfs', 'TRM_outflow_cfs', 'SCC_outflow_cfs', 'ISB_outflow_cfs']]

Sbaseline = df_baseline.loc[:,['SHA_storage_af','ORO_storage_af','BUL_storage_af','FOL_storage_af','PAR_storage_af','NHG_storage_af','NML_storage_af','DNP_storage_af','EXC_storage_af','MIL_storage_af', 'PNF_storage_af', 'TRM_storage_af', 'SCC_storage_af', 'ISB_storage_af']]
Sopt = df_opt.loc[:,['SHA_storage_af','ORO_storage_af','BUL_storage_af','FOL_storage_af','PAR_storage_af','NHG_storage_af','NML_storage_af','DNP_storage_af','EXC_storage_af','MIL_storage_af', 'PNF_storage_af', 'TRM_storage_af', 'SCC_storage_af', 'ISB_storage_af']]



#%% save parameters for plotting comparison
names = ['SHA','ORO','BUL','FOL','PAR','NHG','NML','DNP','EXC','MIL','PNF','TRM','SCC','ISB']

x = np.zeros((14,3))

# parameter 1: # days to allow zero risk (0-10)
# parameter 2: slope of risk/day allowed after that (0-0.1)
# parameter 3: multiplier to change tocs (0.8-1.4)

for k in range(0,14):
    x[k,0] = params_in[k][-3]
    x[k,1] = params_in[k][-2]
    x[k,2] = params_in[k][-1]

params_df = pd.DataFrame(data = x,
                          columns = ['P1','P2','P3'],
                          index = names)
params_df = params_df.reset_index()
params_df = params_df.rename(columns={"index":"Reservoir"})

#if forecast == 'CNRFC':
    #params_df.to_csv("data/optimized_params.csv")
#else:
    #params_df.to_csv("data/perfect_params.csv")

# #%%
Delta = np.ones(14)
max_release = max_release.astype(np.float64)

res = model.objective(Rbaseline.values, Sbaseline.values, Delta, Kr, max_release)
print('The baseline (no forecast) objective function value is ' +str(res))

res = model.objective(Ropt.values, Sopt.values, Delta, Kr, max_release)
print('The optimized objective function value is ' +str(res))

# plot the new policy compared to the old
params_loocv = json.load(open('data/params_firo_loocv.json'))
blank = np.zeros((14,4))
params_loocv_df = pd.DataFrame(data = blank,
                            columns=['Reservoir','P1','P2','P3'])
for i, name in enumerate(names):
    
    params_loocv_df.iloc[i,0] = name
    params_loocv_df.iloc[i,1] = params_loocv[name][7]
    params_loocv_df.iloc[i,2] = params_loocv[name][8]
    params_loocv_df.iloc[i,3] = params_loocv[name][9]

opt_params = pd.read_csv('data/optimized_params.csv')
opt_params = opt_params.drop(['Unnamed: 0'], axis=1)

plt.rcParams['figure.figsize'] = [8, 6]

plt.subplot(3,1,1)
plt.scatter(opt_params.iloc[:,0], opt_params.iloc[:,1], color = 'g')
plt.scatter(params_loocv_df.iloc[:,0], params_loocv_df.iloc[:,1], color = 'r')
plt.title('Lead days with no allowable risk')
plt.ylim(0,10)

plt.subplot(3,1,2)
plt.scatter(opt_params.iloc[:,0], opt_params.iloc[:,2], color = 'g')
plt.scatter(params_loocv_df.iloc[:,0], params_loocv_df.iloc[:,2], color = 'r')
plt.ylim(0,0.1)
plt.title('Slope of risk curve')

plt.subplot(3,1,3)
plt.scatter(opt_params.iloc[:,0], opt_params.iloc[:,3], color = 'g')
plt.scatter(params_loocv_df.iloc[:,0], params_loocv_df.iloc[:,3], color = 'r')
plt.legend(['Original','LOOCV'], loc='lower right')
plt.ylim(1.0,2.0)
plt.title('TOCS Multiplier')

plt.tight_layout()
plt.show()
#%%
for i,name in enumerate(names):
    plt.plot(df_hist[name+'_inflow_cfs'], label = name)
plt.legend()
plt.ylabel('Inflow (cfs)')
plt.xlabel('Date')
plt.title('Observed inflow by site')
plt.show()


# find index of wettest year
start = pd.to_datetime('2017-10-01')
end = pd.to_datetime('2018-09-30')

start_index = df_hist.index.get_loc(start)
end_index = df_hist.index.get_loc(end)
print(start_index)
print(end_index)

print(df_hist.iloc[1430,:])

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

forecast = 'CNRFC'
if forecast == 'CNRFC':
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

if forecast == 'CNRFC':
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

if forecast == 'CNRFC':
    params_df.to_csv("data/optimized_params.csv")
else:
    params_df.to_csv("data/perfect_params.csv")

Delta = np.ones(14)
max_release = max_release.astype(np.float64)

res = model.objective(Rbaseline.values, Sbaseline.values, Delta, Kr, max_release)
print('The baseline (no forecast) objective function value is ' +str(res))

res = model.objective(Ropt.values, Sopt.values, Delta, Kr, max_release)
print('The optimized objective function value is ' +str(res))

#%% Tulare troubleshooting

# change the FIRO params in line

print(Sbaseline[test+'_storage_af'].mean())

print(Rbaseline[test+'_outflow_cfs'].mean())

if Ropt[test+'_outflow_cfs'].max() > nodes[test]['safe_release_cfs']:
    excess = Ropt[test+'_outflow_cfs'].max() - nodes[test]['safe_release_cfs']
    print("Max release exceeded by " +str(excess))
else:
    print("Max release not exceeded")

#%%

# for i in range(len(names)):
#     name = names[i]
#     print(name)
#     print(params[name][4]*params[name][9])
#     print(params[name][4])
# #%%
# for i in range(len(names)):
#     name = names[i]
#     print(name)
#     print(df_opt[name+'_storage_af'].mean())
#     print(df_baseline[name+'_storage_af'].mean())


# #%%

# for i in range(len(names)):
#     name = names[i]
#     plt.plot(df_opt[name+'_tocs_fraction'])
#     plt.plot(df_baseline[name+'_tocs_fraction'])
#     plt.title(name)
#     plt.legend(['Opt','Baseline'])
#     plt.show()
#     print(name)
#     print(df_hist[name+'_inflow_cfs'].std()/max_release[i])

#%%

# day1 = model.reservoir_step(
#     params_in[10], #rule param
#     94, #dowy
#     df_opt.loc['2017-01-03', 'PNF_outflow_cfs'], #inflow Q
#     df_opt.loc['2017-01-03', 'PNF_storage_af'], #storage S
#     Kr[10], #capacity K
#     medians.iloc[94, medians.columns.get_loc('PNF_outflow_cfs')], # median release R_avg
#     medians.iloc[94, medians.columns.get_loc('PNF_storage_af')], # median storage S_avg
#     df_opt.loc['2017-01-03', 'PNF_tocs_fraction'], #tocs
#     max_release[10], #max safe release
#     #use_firo=True,
#     forecasts, #forecasts
#     risks)

#(x, dowy, Q, S, K, R_avg, S_avg, tocs, R_max, use_firo=False, Qf=np.zeros((40,14)), risk_threshold=np.zeros(14))

#Advances reservoir storage from one timestep to the next

  # Parameters:
  #   x (np.array): Reservoir rule parameters (7 + 2 firo)
  #   dowy (int): Day of water year
  #   Q (float): Inflow, cfs
  #   S (float): Storage, acre-feet
  #   K (float): Storage capacity, acre-feet
  #   R_avg (float): Median release for this day of the year, cfs
  #   S_avg (float): Median storage for this day of the year, acre-feet
  #   tocs (float): Top of conservation storage, fraction of capacity
  #   R_max (float): max safe release, cfs
  #   use_firo (boolean, default=False): Option to use FIRO policy
  #   Qf (np.array(float, float), default=np.zeros((40,14))): Inflow forecast ensemble, cfs (ens, lead)
  #   risk_threshold (np.array, default=np.zeros(14)): Piecewise linear curve for allowable risk at each lead time, unitless

#%% plotting

#mean normalized (by capacity) difference storage vs mean difference in tocs

# S_diff = np.zeros(14)
# tocs_diff = np.zeros(14)


# for i in range(len(names)):
#     name = names[i]
#     S_diff[i] = (df_opt[name+'_storage_af'].mean() - df_baseline[name+'_storage_af'].mean())/Kr[i]
#     tocs_diff[i] = (df_opt[name+'_tocs_fraction'].mean()*params_in[10][9] - df_baseline[name+'_tocs_fraction'].mean())
    

# for i in range(len(names)):
#     name = names[i]
#     plt.scatter(tocs_diff[i],S_diff[i], s=(Kr[i])**0.5)
#     plt.annotate(name, (tocs_diff[i], S_diff[i]))
#     plt.xlabel('Mean tocs delta')
#     plt.ylabel('Mean storage delta (normalized)')
# plt.title('Optimized and Baseline tocs comparison')
# plt.show()

# #%%

# for i in range(len(names)):
#     name = names[i]
#     plt.scatter(S_diff[i],max_release[i]/Kr[i])
#     plt.annotate(name, (S_diff[i], max_release[i]/Kr[i]))
#     plt.xlabel('Mean tocs delta')
#     plt.ylabel('Mean storage delta (normalized)')
# plt.title('Optimized and Baseline tocs comparison')
# plt.show()


# #%%

# for i in range(len(names)):
#     name = names[i]
#     two_week_rolling = df_hist[name+'_inflow_cfs'].rolling(window=30).sum()
#     highest = two_week_rolling.idxmax()
#     plt.scatter(tocs_diff[i], two_week_rolling[highest]*1.983/Kr[i], s=(Kr[i]**0.5))
#     plt.annotate(name, (tocs_diff[i], two_week_rolling[highest]*1.983/Kr[i]))
#     plt.ylabel('Rolling wettest month normalized by capacity')
#     plt.xlabel('Tocs delta')
# plt.title('Relationship between tocs delta and wettest month')
# plt.show()    


# #%%

# for i in range(len(names)):
#     name = names[i]
#     ratio = df_hist[name+'_inflow_cfs'].max()/max_release[i]/(Kr[i])*1000
#     plt.scatter(tocs_diff[i], ratio)
#     plt.annotate(name, (tocs_diff[i], ratio))
#     plt.xlabel('Tocs Diff')
#     plt.ylabel('Ratio of Max inflow to max release (normalized)')
#     plt.title('Relationship between tocs delta and max inflow/outflow ratio')
# plt.show()
    
#%%

for i in range(len(names)):
    name = names[i]
    ramp = ramping_rate[i]
    temp = Ropt[name+'_outflow_cfs']
    temp = pd.DataFrame(temp)
    temp['rate_of_change'] = temp[name+'_outflow_cfs'].diff()
    max_rate = temp['rate_of_change'].max()
    print(name)
    print(max_rate)
    print(ramp)
    if max_rate > ramp:
        print("Ramping Rate Exceeded!!!")
    else:
        print("we good")

#%%

for i in range(len(names)):
    name = names[i]
    print(name)
    print(Sbaseline[name+'_storage_af'].mean())
    print(Sopt[name+'_storage_af'].mean())

#%%

for i in range(len(names)):
    name = names[i]
    plt.plot(df_baseline[name+'_tocs_fraction'])
plt.show()

for i in range(len(names)):
    name = names[i]
    plt.plot(df_opt[name+'_tocs_fraction'])
plt.show()

#%%
start_year = 2013
end_year = 2023
for year in range(start_year, end_year+1):
    
    start_exclude = pd.to_datetime(f"{year}-10-01")
    end_exclude = start_exclude + timedelta(days=29)
    
    filtered_data = df_baseline[(df_baseline.index < start_exclude) | (df_baseline.index > end_exclude)]
    

    
#%% Compare median releases to historical
res = 'SHA'
df = pd.DataFrame(data=df_hist[res+'_outflow_cfs'])
df.index = pd.to_datetime(df.index)

def assign_water_year(date):
    if date.month >= 10:
        return date.year + 1  # October, November, December belong to the next water year
    else:
        return date.year

# Apply this function to the DataFrame index to create a water year column
df['water_year'] = df.index.map(assign_water_year)

fig, ax = plt.subplots(figsize=(10,6))

water_years = df['water_year'].unique()

water_years = df['water_year'].unique()

for i, water_year in enumerate(water_years):
    # Select only the release values for the current water year
    yearly_data = df[df['water_year'] == water_year][res+'_outflow_cfs'].values
    
    # Generate a sequence of days for the x-axis, assuming daily data
    days = np.arange(len(yearly_data))
    
    # Plot the data
    ax.plot(days, yearly_data, label=str(water_year), alpha = 0.5)


ax.plot(medians[res+'_outflow_cfs'].values, label = 'Historical Median', c='black')

ax.set_xlabel('Day of water year')
ax.set_ylabel('Release (cfs)')
ax.legend(title='Water Year')
ax.set_title('Shasta Release Values Comparison Across Water Years')
plt.xticks(rotation=45)  # Rotate x-tick labels for better readability

plt.show()


# how many days was demand met?
count = 0
count_sum = 0
for water_year in water_years:
    
    yearly_data = df[df['water_year'] == water_year][res+'_outflow_cfs'].values
    
    min_length = min(len(yearly_data), len(medians))
    
    for j in range(min_length):
        if yearly_data[j] > medians.iloc[j, 1]:
            count += 1
    count_sum += count
    print(f"Count for Water Year {water_year}: {count}")
    count = 0

print(count_sum/len(water_years))


#%% plot the inflows

for i,name in enumerate(names):
    plt.plot(df_hist[name+'_inflow_cfs'], label = name)
plt.legend()
plt.show()


# find index of wettest year
start = pd.to_datetime('2017-10-01')
end = pd.to_datetime('2018-09-30')

start_index = df_hist.index.get_loc(start)
end_index = df_hist.index.get_loc(end)
print(start_index)
print(end_index)

print(df_hist.iloc[1430,:])




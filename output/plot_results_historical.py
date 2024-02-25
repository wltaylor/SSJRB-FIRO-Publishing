import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import json
import datetime

dfo = pd.read_csv('../data/historical.csv', index_col=0, parse_dates=True)['11-01-2013':]
dfs = pd.read_csv('sim_baseline_og_params.csv', index_col=0, parse_dates=True)
variables = json.load(open('../data/nodes.json'))
rk = [k for k in variables.keys() if (variables[k]['type'] == 'reservoir') and variables[k]['fit_policy']]

fig,ax = plt.subplots(nrows=len(rk), sharex=True)

for i,r in enumerate(rk):
  k = r+'_storage_af'
  dfo[k].plot(ax=ax[i])
  dfs[k].plot(ax=ax[i])
  r2 = np.corrcoef(dfo[k].values, dfs[k].values)[0,1]**2
  ax[i].set_title('%s Storage: R2=%0.2f' % (r,r2))
  plt.legend(['Obs','Sim'])

plt.show()

fig,ax = plt.subplots(nrows=len(rk), sharex=True)

for i,r in enumerate(rk):
  k = r+'_outflow_cfs'
  dfo[k].plot(ax=ax[i])
  dfs[k].plot(ax=ax[i])
  r2 = np.corrcoef(dfo[k].values, dfs[k].values)[0,1]**2
  ax[i].set_title('%s Release: R2=%0.2f' % (r,r2))
  plt.legend(['Obs','Sim'])

plt.show()

ks = ['delta_gains_cfs', 'delta_inflow_cfs', 'HRO_pumping_cfs', 'TRP_pumping_cfs', 'total_delta_pumping_cfs', 'delta_outflow_cfs']
fig,ax = plt.subplots(nrows=len(ks), sharex=True)

for i,k in enumerate(ks):
  dfo[k].plot(ax=ax[i])
  dfs[k].plot(ax=ax[i])
  r2 = np.corrcoef(dfo[k].values, dfs[k].values)[0,1]**2
  ax[i].set_title('%s: R2=%0.2f' % (k,r2))
  plt.legend(['Obs','Sim'])

plt.show()



#%% Will version

dfb = pd.read_csv('../data/baseline_release.csv', header=None)
dfo = pd.read_csv('../data/historical.csv', index_col = 0, parse_dates=True)['11-01-2013':]
dfs = pd.read_csv('sim_baseline_og_params.csv', index_col=0, parse_dates=True)

start_date = datetime.datetime.strptime('11-01-2013', "%m-%d-%Y")

# initializing K
K = len(dfb)
dates = pd.date_range(start_date, periods=K)

names = ['SHA','ORO','BUL','FOL','PAR','NHG','NML','DNP','EXC','MIL','PNF','TRM','SCC','ISB']

dfb = pd.DataFrame(data = dfb.values,
                  index = dates,
                  columns = names)

fig,ax = plt.subplots(nrows = len(names), sharex=True)

for i in range(len(names)):
    name = names[i]
    dfb[name].plot(ax = ax[i])
    dfo[name+'_outflow_cfs'].plot(ax = ax[i])
    r2 = np.corrcoef(dfo[name+'_outflow_cfs'].values, dfb[name].values)[0,1]**2
    ax[i].set_title('%s: R2=%0.2f' % (name,r2))
    plt.legend(['Baseline','Obs'])
plt.tight_layout()
plt.show()



# compare the R2 values
original_r2 = np.zeros(len(names))
new_r2 = np.zeros(len(names))

for i in range(len(names)):
    name = names[i]
    original_r2[i] = np.corrcoef(dfo[name+'_outflow_cfs'].values, dfs[name+'_outflow_cfs'].values)[0,1]**2
    new_r2[i] = np.corrcoef(dfo[name+'_outflow_cfs'].values, dfb[name].values)[0,1]**2

#combine
r2_df = pd.DataFrame({
    'Original': original_r2,
    'New': new_r2})
r2_df.index = names

plt.rcParams['figure.figsize'] = [8, 6]

ax = r2_df.plot(kind='bar', width=0.5, position=1, align='center', color = ['r','b'])
plt.xlabel('Reservoir')
plt.ylabel('R^2')
plt.title('R^2 values between simulated and observed releases')

    
    
    
    
    
    
    
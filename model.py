import numpy as np 
import matplotlib.pyplot as plt
from numba import njit
from util import *

'''
SSJRB simulation model. Three parts:
Reservoir releases, gains (into Delta), and Delta pumping
Each component has two functions: step() and fit()
Then all components are combined in the simulate() function
Numba compilation requires only numpy arrays, no pandas objects
'''

@njit
def get_tocs(x, d):
  tp = [0, x[1], x[2], x[3], 366]
  sp = [1, x[4], x[4], 1, 1]
  return np.interp(d, tp, sp)

@njit
def get_risk_thresholds(x):
  # find risk threshold - piecewise linear function
  nl = 14
  risk_threshold = np.zeros(nl)
  for l in range(nl):
    if l > int(x[7]):
      risk_threshold[l] = x[8] * (l - int(x[7]))

  return risk_threshold


@njit
def firo_policy(x, S, Qf, Q, tocs, K, risk_threshold):
  '''
  Implements FIRO risk threshold policy
    Parameters:
      x (np.array): Reservoir rule parameters (7 + 2 firo)
      S (float): Storage, acre-feet
      Qf np.array(float, float): Inflow forecast ensemble, cfs (ens, lead)
      tocs (float): Top of conservation storage, fraction of capacity
      K (float): Storage capacity, acre-feet

    Returns:
      float: Target release for this policy
  '''

  nl = 14 # number of leads
  n_ens = 40 
  R = np.zeros(nl)

  for l in range(nl):
    ix = int((1-risk_threshold[l]) * (n_ens-1))
    Qf_q = Qf[ix,:(l+1)].sum() * cfs_to_afd # inflow forecast with exceedance % = risk threshold %
    
    if ~np.isnan(Qf_q):
      Sf_q = S + Qf_q # storage forecast
    else:
      # no forecasts available in early years for some reservoirs
      # in this case use the regular flood pool with one-day inflows
      Sf_q = S + Q

    if Sf_q > K * tocs:
      #R[l] = (Sf_q - K * tocs) * x[5] / (l+1) # if tocs exceeded, distribute releases over l days
      R[l] = (Sf_q - K * tocs) / (l+1)
  return np.max(R) # maximum release should maintain risk threshold for all lead times

@njit 
def reservoir_step(x, dowy, Q, S, K, R_avg, S_avg, tocs, R_max, Ramp, R_previous, use_firo=False, Qf=np.zeros((40,14)), risk_threshold=np.zeros(14)):
  '''
  Advances reservoir storage from one timestep to the next

    Parameters:
      x (np.array): Reservoir rule parameters (7 + 2 firo)
      dowy (int): Day of water year
      Q (float): Inflow, cfs
      S (float): Storage, acre-feet
      K (float): Storage capacity, acre-feet
      R_avg (float): Median release for this day of the year, cfs
      S_avg (float): Median storage for this day of the year, acre-feet
      tocs (float): Top of conservation storage, fraction of capacity
      R_max (float): max safe release, cfs
      Ramp (float): ramping rate, cfs
      R_previous: previous timestep release value, cfs
      use_firo (boolean, default=False): Option to use FIRO policy
      Qf (np.array(float, float), default=np.zeros((40,14))): Inflow forecast ensemble, cfs (ens, lead)
      risk_threshold (np.array, default=np.zeros(14)): Piecewise linear curve for allowable risk at each lead time, unitless
      
      
    Returns:
      tuple(float, float): the updated release (cfs) and storage (af)

  '''
  R_avg *= cfs_to_afd
  Q *= cfs_to_afd
  # do not modify Qf here. it's a np array passed by reference

  R_target = R_avg # default

  if S < S_avg:
    R_target = R_avg * (S / S_avg) ** x[0] # exponential hedging

  # if using firo policy, the flood pool is only used when Sf > tocs
  if use_firo:
    R_firo = firo_policy(x, S, Qf, Q, tocs, K, risk_threshold)
    if R_firo > R_target:
      R_target = R_firo

  # otherwise in the baseline case the flood pool is always used
  # todo make this an input option in model.simulate
  S_target = S + Q - R_target 
  if S_target > K * tocs: 
    R_target += (S_target - K * tocs) #* x[5]

  # limit to the safe release if there is no spill
  R_target = np.min(np.array([R_target, R_max * cfs_to_afd]))
  
  # limit to the ramping rate, need to account for ramping up or down
  temp_ramp = R_previous + Ramp # ramp up limit
  R_target = np.min(np.array([R_target, temp_ramp * cfs_to_afd]))
  
  temp_ramp = R_previous - Ramp # ramp down limit
  R_target = np.max(np.array([R_target, temp_ramp * cfs_to_afd]))

  S_target = S + Q - R_target 

  if S_target > K: # spill can cause R to exceed R_max
    R_target += S_target - K
  elif S_target < K * x[6]: # below dead pool
    R_target -= (K * x[6] - S_target)

  R_target = np.max(np.array([R_target, 0])) 
  S_target = np.max(np.array([S + Q - R_target, 0]))

  return (R_target * afd_to_cfs, S_target)


@njit
def reservoir_fit(x, dowy, Q, K, Q_avg, R_avg, R_obs, S_avg, S_obs, S0, R_max, Ramp):
  '''
  Evaluate reservoir model against historical observations for a set of parameters 

    Parameters:
      x (np.array): Reservoir rule parameters (5)
      dowy (np.array(int)): Day of water year over the simulation
      Q (np.array(float)): Inflow, cfs
      S (np.array(float)): Storage, acre-feet
      K (float): Storage capacity, acre-feet
      R_avg (np.array(float)): Median release for each day of the year, cfs
      R_obs (np.array(float)): Observed historical release, cfs
      S_avg (np.array(float)): Median storage for each day of the year, acre-feet
      S_obs (np.array(float)): Observed historical storage, acre-feet. 
        Not used currently, but could fit parameters to this instead.
      S0 (float): initial storage, acre-feet
      R_max (float): safe downstream storage

    Returns:
      (float): the negative r**2 value of reservoir releases, to be minimized

  '''
  T = dowy.size
  R,S = np.zeros(T), np.zeros(T)
  S[0] = S0
  tocs = get_tocs(x, dowy)

  for t in range(1,T):
    inputs = (dowy[t], Q[t], S[t-1], K, R_avg[dowy[t]], S_avg[dowy[t-1]], tocs[t], R_max, Ramp, R[t-1])
    R[t], S[t] = reservoir_step(x, *inputs)
    
  return -np.corrcoef(S_obs, S)[0,1]**2


@njit
def gains_step(x, dowy, Q_total, Q_total_avg, S_total_pct, Gains_avg):
  '''
  Compute gains into the Delta for one timestep

    Parameters:
      x (np.array): Gains parameters 
      dowy (int): Day of water year
      Q_total (float): Total inflow to all reservoirs, cfs
      Q_total_avg (float): Average total inflow for this day of the year, cfs
      S_total_pct (float): System-wide reservoir storage, % of median
      Gains_avg (float): Median gains for this day of the year, cfs

    Returns:
      (float): Gains into the Delta for one timestep, cfs

  '''
  G = Gains_avg * S_total_pct ** x[0] # adjust up/down for wet/dry conditions
  if Q_total > x[1] * Q_total_avg: # high inflows correlated with other tributaries
    G += Q_total * x[2]
  return G


@njit
def gains_fit(x, dowy, Q_total, Q_total_avg, S_total_pct, Gains_avg, Gains_obs):
  '''
  Evaluate Delta gains model against historical observations for a set of parameters

    Parameters:
      x (np.array): Gains parameters
      dowy (np.array(int)): Day of water year
      Q_total (np.array(float)): Total inflow to all reservoirs, cfs
      Q_total_avg (np.array(float)): Average total inflow 
                                     for this day of the year, cfs
      S_total_pct (np.array(float)): System-wide reservoir storage, % of median
      Gains_avg (np.array(float)): Median gains for each day of the year, cfs
      Gains_obs (np.array(float)): Observed historical gains, cfs

    Returns:
      (float): the negative r**2 value of Delta gains, to be minimized

  '''
  T = dowy.size
  G = np.zeros(T)

  for t in range(T):
    inputs = (dowy[t], Q_total[t], Q_total_avg[dowy[t]], S_total_pct[t], Gains_avg[dowy[t]])
    G[t] = gains_step(x, *inputs)

  return -np.corrcoef(Gains_obs, G)[0,1]**2


@njit
def pump_step(x, dowy, Q_in, Kp, Pump_pct_avg, Pump_avg, S_total_pct):
  '''
  Compute Delta pumping for one timestep

    Parameters:
      x (np.array): Delta pumping parameters
      dowy (int): Day of water year
      Q_in (float): Total inflow to the Delta, cfs 
                    (sum of all reservoir outflows plus gains)
      Kp (float): Pump capacity, cfs
      Pump_pct_avg (float): (Average pumping / Average inflow) 
                    for this day of the year, unitless
      S_total_pct (float): System-wide reservoir storage, % of median

    Returns:
      tuple(float, float): Pumping (cfs) and outflow (cfs)

  '''

  Q_in = np.max(np.array([Q_in, 0.0]))

  export_ratio = x[5] if dowy < 273 else x[6]
  outflow_req = x[3] if dowy < 273 else x[4]

  P = Q_in * Pump_pct_avg * S_total_pct ** x[0] # ~ export ratio

  if Q_in < x[1]: # approximate dry year adjustment
    P *= np.min(np.array([S_total_pct, 1.0])) ** x[2]

  if P > Q_in * export_ratio:
    P = Q_in * export_ratio
  if P > Q_in - outflow_req:
    P = np.max(np.array([Q_in - outflow_req, 0]))

  if 182 <= dowy <= 242: # env rule apr-may
    Kp = 750
  
  P = np.min(np.array([P, Kp]))
  return P


@njit
def pump_fit(x, dowy, Q_in, Kp, Pump_pct_avg, Pump_cfs_avg, S_total_pct, Pump_obs):
  '''
  Evaluate pump policy against historical observations for a set of parameters

    Parameters:
      x (np.array): Delta pumping parameter (1)
      dowy (np.array(int)): Day of water year
      Q_in (np.array(float)): Total inflow to the Delta, cfs 
                              (sum of all reservoir outflows plus gains)
      Kp (float): Pump capacity, cfs
      Pump_pct_avg (np.array(float)): (Average pumping / Average inflow) 
                                      for each day of the year, unitless
      S_total_pct (np.array(float)): System-wide reservoir storage, % of median
      Pump_obs (np.array(float)): Observed historical pumping, cfs

    Returns:
      (float): the negative r**2 value to be minimized

  '''
  T = dowy.size
  P = np.zeros(T)

  for t in range(T):
    inputs = (dowy[t], Q_in[t], Kp, Pump_pct_avg[dowy[t]], Pump_cfs_avg[dowy[t]], S_total_pct[t])
    P[t] = pump_step(x, *inputs)

  return -np.corrcoef(Pump_obs, P)[0,1]**2


@njit
def simulate(params, Kr, Kp, dowy, Q, Q_avg, R_avg, S_avg, Gains_avg, Pump_pct_avg, Pump_avg, DM, S0, R_max, Ramp, use_firo=False, Qf=None):
  '''
  Run full system simulation over a given time period.

    Parameters:
      params (tuple(np.array)): Parameter arrays for all reservoirs, gains, and Delta
      Kr (np.array(float)): Reservoir capacities, af
      Kp (np.array(float)): Pump capacities, cfs
      dowy (np.array(int)): Day of water year
      Q (np.array(float, float)): Matrix of inflows at all reservoirs, cfs
      Q_avg (np.array(float, float)): Matrix of median inflows for each reservoir
                                      for each day of the year, cfs
      R_avg (np.array(float, float)): Matrix of median releases for each reservoir
                                      for each day of the year, cfs
      S_avg (np.array(float, float)): Matrix of median storage for each reservoir
                                      for each day of the year, af
      Gains_avg (np.array(float)): Median gains for each day of the year, cfs
      Pump_pct_avg (np.array(float, float)): (Median pumping / median inflow) 
                                             for each day of the year for each reservoir, unitless
      Pump_avg (np.array(float, float)): Median pumping for each day of the year
                                         for each reservoir, cfs
      DM (np.array(float)): Demand multiplier, system-wide, unitless
                            (=1.0 for the historical scenario)
      S0 (np.array(float)): Initial storage for each reservoir, af
      R_max (np.array(float)): Safe downstream release for each reservoir, cfs
      use_firo (bool, default=False): Option to use FIRO policy
      Qf (np.array (float, float, float, float), default=None): Forecast ensemble matrix with dimensions (site, time, ens, lead)

    Returns:
      (tuple(np.array, np.array, np.array)): Matrices of timeseries results (cfs) 
        for reservoir releases, storage, and Delta Gains/Inflow/Pumping/Outflow

  '''  
  T,NR = Q.shape # timesteps, reservoirs
  NP = 2 # pumps
  NL = 14 # lead time days
  R,S,G,I,P = (np.zeros((T,NR)), np.zeros((T,NR)), 
               np.zeros(T), np.zeros(T), np.zeros((T,NP)))
  Q_total = Q.sum(axis=1)
  Q_total_avg = Q_avg.sum(axis=1)
  S_total_avg = S_avg.sum(axis=1)
  S[0,:] = S0
  R[0,:] = 0 # needed to add an initial release value

  tocs = np.zeros((T,NR))
  risk_thresholds = np.zeros((NR,NL))

  for r in range(NR):
    tocs[:,r] = get_tocs(params[r], dowy)

  if use_firo:
    for r in range(NR):
      tocs[:,r] *= params[r][9]
      risk_thresholds[r,:] = get_risk_thresholds(params[r])
    risk_thresholds = np.clip(risk_thresholds, 0, 1) # valid quantile values
    tocs = np.clip(tocs, 0, 1) # valid tocs fractions

  for t in range(1,T):

    d = dowy[t]

    # 1. Reservoir policies
    for r in range(NR):
      res_demand = R_avg[d,r] * DM[t] # median historical release * demand multiplier
      inputs_r = (d, Q[t,r], S[t-1,r], Kr[r], res_demand, S_avg[d,r], tocs[t,r], R_max[r], Ramp[r], R[t-1,r])

      if use_firo and Qf is not None:
        R[t,r], S[t,r] = reservoir_step(params[r], *inputs_r, use_firo, Qf[r,t], risk_thresholds[r])
      elif use_firo and Qf is None:
        raise ValueError('if use_firo is True, Qf must be defined')
      else:
        R[t,r], S[t,r] = reservoir_step(params[r], *inputs_r)

    # 2. Gains into Delta
    S_total_pct = S[t].sum() / S_total_avg[d]
    inputs_g = (d, Q_total[t], Q_total_avg[d], 
              S_total_pct, np.min(np.array([Gains_avg[d] * DM[t], Gains_avg[d]])))
    G[t] = gains_step(params[NR], *inputs_g)

    # 3. Delta pumping policies
    I[t] = R[t].sum() + G[t]
    for p in range(NP):
      if p == 0: S_total_pct = S[t,1] / S_avg[d,1] # SWP - ORO only
      inputs_p = (dowy[t], I[t], Kp[p], Pump_pct_avg[d,p], Pump_avg[d,p], S_total_pct)
      P[t,p] = pump_step(params[NR+p+1], *inputs_p)

  Delta = np.vstack((G,I,P[:,0],P[:,1],I-P.sum(axis=1))).T # Gains, Inflow, Pumping, Outflow
  return (R, S, Delta, tocs)


@njit
def objective(R, S, Delta, Kr, max_release):
  NR = len(Kr)
  obj = 0
  
  for r in range(NR):
    S_norm = S[:,r] / Kr[r]
    obj -= S_norm.mean() / NR # maximize average storage
    #obj += np.sum(S_norm > 0.99) * 10 # large overtopping penalty

    obj += np.sum(R[:,r] > max_release[r]) * 1 # penalty for any releases > safe limit
  return obj

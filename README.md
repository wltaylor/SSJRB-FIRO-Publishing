#### Sacramento-San Joaquin River Basin (SSJRB) simulation model
#### FIRO Project


**Requirements:** [NumPy](http://www.numpy.org/), [pandas](http://pandas.pydata.org/), [Matplotlib](http://matplotlib.org/), [SciPy](https://www.scipy.org/) (optional), [Numba](http://numba.pydata.org/) (optional)

#### Steps:

#### Data preparation (optional): `data_update.py`
* Network components are defined in `data/nodes.json`
* Historical data can be updated to the current day from [CDEC](https://cdec.water.ca.gov/). Run with `updating=True`.
* CNRFC HEFS forecasts are downloaded for all regions, extracted for these sites, out to a 14-day lead time. 

#### Fit parameters (optional): `fit_historical.py`
* Fits values for reservoir policies, gains, and delta pumping over the historical period. Saved in `data/params.json`.

#### Simulation: `simulate_firo.py`
* Simulates baseline (non-FIRO) policy, saves output `output/sim_historical.csv` to plot goodness of fit using `output/plot_results_historical.py`
* Simulates most and least risk-averse FIRO policies, as defined by the space of three parameters for each reservoir. 
* To use FIRO option, specify `model.simulate(..., use_firo=True)`. This requires also specifying the forecasted inflows `Qf`, which is a 4-dimensional numpy array with dimensions `(site, time, ensemble member, lead time)`. 

#### Optimization: `train_firo.py`
* Optimizes 3 FIRO parameters per reservoir to maximize the gain in system-wide storage as a percent of capacity
* Large penalty added for flood releases exceeding the max safe release defined in `nodes.json` for each reservoir. This ensures that optimizing the storage gain does not increase the flood risk.
* The optimized policy parameters are added to the fitted historical parameters and saved in `data/params_firo.json`. These can be read in and simulated with a setup similar to `simulate_firo.py`.

#### future todo
* add ramping rates or infiltration limits to decide releases in `model.firo_policy()`. Right now it uses the average volume over the lead time to stay below tocs.
* how to quantify benefit in the objective function

#### License: [MIT](LICENSE.md)

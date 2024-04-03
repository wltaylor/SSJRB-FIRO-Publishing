## Sacramento-San Joaquin River Basin (SSJRB) simulation model
Files supporting the manuscript 'Variability, attributes, and drivers of optimal forecast-informed reservoir operating policies for water supply and flood control in California' submitted 2 April 2024.

**Requirements:** [NumPy](http://www.numpy.org/), [pandas](http://pandas.pydata.org/), [Matplotlib](http://matplotlib.org/), [SciPy](https://www.scipy.org/), [Xarray](https://docs.xarray.dev/en/stable/) [Numba](http://numba.pydata.org/) (optional)

## Steps:

#### Data preparation (optional):
* Network components are defined in `data/nodes.json`
* Historical data can be updated to the current day from [CDEC](https://cdec.water.ca.gov/).
* CNRFC HEFS forecasts are downloaded for all regions, extracted for these sites, out to a 14-day lead time, located in the linked folder: [Box](https://ucdavis.box.com/s/hj1q0o8xhqrr6dyhxyreqwrkqzhupfbw). HEFS from other time periods are available at [CNRFC](https://www.cnrfc.noaa.gov/).

#### Fit parameters (optional): `fit_historical.py`
* Fits values for reservoir policies, gains, and delta pumping over the historical period. Saved in `data/params.json`.

#### Optimization: `train_firo.py`
* Optimizes 3 FIRO parameters per reservoir to maximize the gain in system-wide storage as a percent of capacity.
* Large penalty added for flood releases exceeding the max safe release defined in `nodes.json` for each reservoir. This ensures that optimizing the storage gain does not increase the flood risk.
* The optimized policy parameters are added to the fitted historical parameters and saved in `data/params_firo.json`. These can be read in and simulated with `simulate_firo.py`.

#### Simulation: `simulate_firo.py` and `simulate_baseline_tocs.py`
* Simulates baseline (non-FIRO) policy, saves output `output/sim_historical.csv` to plot goodness of fit using `output/plot_results_historical.py`
* To use FIRO option, specify `model.simulate(..., use_firo=True)`. This requires also specifying the forecasted inflows `Qf`, which is a 4-dimensional numpy array with dimensions `(site, time, ensemble member, lead time)`. `Qf` can be changed to the perfect forecasts to replicate the results in the manuscript.
* `simulate_baseline_tocs.py` uses the baseline rules but iteratively increases the TOCS value until a constraint (spill or max release exceedance) is broken. The results are saved in  `data` and can be compared to the other time series in `analysis.py`.

#### Leave one out cross validation: `loocv.py`
* Identical to `train_firo.py` except that unique water years are masked in order to create held out policies. The policies are saved in the `data` folder and can be used to generate time series results for each reservoir using `simulate_loocv.py`.

#### Figure Generation and Analysis: `analysis.py`
* Creates the figures shown in the manuscript.

#### Contact
William Taylor wltaylor@ucdavis.edu

#### License: [MIT](LICENSE.md)

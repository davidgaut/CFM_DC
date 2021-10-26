# CFM_DC
Codes for 2021 CFM-Data challenge

The estimation is started with CFM_challenge.py

  - CFM_options.json contains all options such as lag/steps/var_to_lags 

  - make_data.py processes .csv files into lgb.datasets with basic feature engineering
  
  - hypertune.py contains the divers functions to tune lgb models based on timeseries cross validation and interface the estimation with Neptune



# %% CFM Data challenge 2021
# Set dir and install modules
import os
os.chdir('/home/davidg/Codes/Python_Codes/Projects/Hackatons/CFM_data_challenge')
# os.system('pip install -r requirements.txt -q')

import CFM_function as cfm
from sklearn.model_selection import TimeSeriesSplit

import json
with open('CFM_options.json') as f:
    options = json.load(f)
    print(json.dumps(options, indent=4, sort_keys=True))
    
#%% Dataset

# If start from scratch
NewEstim = True

if NewEstim:
  train_dataset, y_test_res, X_test, X_idx = cfm.make_dataset(options)
else:
  import pickle
  print('Loading inputs from last run...')
  with open('last_run\\'+'train_dataset.pkl', 'rb') as file: train_dataset = pickle.load(file)
  with open('last_run\\'+'X_idx.pkl', 'rb')         as file: X_idx         = pickle.load(file)
  with open('last_run\\'+'model_ols.pkl', 'rb')     as file: model_ols     = pickle.load(file)
  with open('last_run\\'+'options.pkl', 'rb')       as file: options       = pickle.load(file)

projects = cfm.init(options)
study, neptune_callback = cfm.log_new(projects,NewEstim,options)
# save last name in corresponding folder

objective = lambda trial: cfm.objective(trial,train_dataset,TimeSeriesSplit(n_splits=options['cv_splits']).split(X_idx),eval(options['default_param']))

study.optimize(objective, 
timeout   = 60 * 60 * 2, 
callbacks = [neptune_callback],
# n_jobs=-1
# pruner=optuna.pruners.MedianPruner(),
)

import pickle
with open('last_run\\'+'X_test.pkl', 'rb')     as file: X_test         = pickle.load(file)
with open('last_run\\'+'y_test_res.pkl', 'rb') as file: y_test_res     = pickle.load(file)

cfm.predict(study,train_dataset,TimeSeriesSplit(n_splits=options['cv_splits']).split(X_idx),y_test_res,X_test,options)
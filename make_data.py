from re import findall
import pandas as pd
import numpy as np
from lightgbm import Dataset
from sklearn.base import BaseEstimator
import stats
from sklearn.decomposition import PCA
    
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        if (col_type != object) and (str(col_type) != 'category'):
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df

def import_data(file):
    """create a dataframe and optimize its memory usage"""
    df = pd.read_csv(file, parse_dates=True, keep_date_col=True)
    df = reduce_mem_usage(df)
    return df

def splat_pid(df):
    '''Create multi-groups with dummy 0 or 1 (useful to reduce #bin for GPU with lgb)'''
    for count, group in enumerate(df.pid.unique()):
        df['grp_'+str(count)] = np.zeros((df.shape[0],),dtype=int)
        for g in [group]:
            df['grp_'+str(count)].where(df['pid']!=g,g,inplace=True)
            df['grp_'+str(count)] = df['grp_'+str(count)].astype('category')
    return df


def add_lags(data,lags,roll,var):
    '''Add columns of lagged variables to a df'''
    ids=data.pid.unique()
    for id_pid in ids:
        tab = data.query('pid=='+str(id_pid))
        for lag in lags:
            lag_val = tab[var].shift(lag).rolling(roll).mean()
            data.loc[tab.index,var+'_L'+str(lag)] = lag_val
    return data.fillna(0.0) 

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

def split_data(train_data,test_data):
    """
    Add day/month separatly for seasonality to a dataframe.
    """
    from math import ceil
    def week_of_month(dt):
        """ Returns the week of the month for the specified date.
        """
        first_day = dt.replace(day=1)
        dom = dt.day
        adjusted_dom = dom + first_day.weekday()
        return int(ceil(adjusted_dom/7.0))

    train_data['day_time'] = pd.to_datetime(train_data['day'], unit='D')
    test_data['day_time']  = pd.to_datetime(test_data['day'], unit='D')

    train_data['months'] = train_data['day_time'].dt.month       - 0
    train_data['weeks']  = pd.Series(list(map(week_of_month,train_data['day_time'].dt.to_pydatetime())),index=train_data.index)
    train_data['days']   = train_data['day_time'].dt.dayofweek   - 0
    test_data['months']  = test_data['day_time'].dt.month        - 0
    test_data['weeks']   = pd.Series(list(map(week_of_month,test_data['day_time'].dt.to_pydatetime())),index=test_data.index)
    test_data['days']    = test_data['day_time'].dt.dayofweek    - 0

    train_data, test_data = train_data.drop(['day_time'], axis=1), test_data.drop(['day_time'], axis=1)

    return train_data, test_data

def make_dataset(options):
    '''Construct and return lightgbm dataset.'''
    load_and_treat = lambda path: \
        (clean_dataset(import_data(path).set_index('ID').fillna(0.0).sort_values(by=['day','pid'])))

    application_train = load_and_treat(options["folder_path"]+'/Data/input_training.csv')
    application_test  = load_and_treat(options["folder_path"]+'/Data/input_test.csv')
    target_train      = import_data(options["folder_path"]+'/Data/output_training_IxKGwDV.csv').set_index('ID').fillna(0.0).sort_index()

    application_train, application_test = split_data(application_train, application_test)

    # Set categorical types
    application_train = application_train.astype({"days": int, "weeks": int, "months": int, "pid": int, "day": int})#.iloc[:5000]
    application_test  = application_test.astype({"days": int, "weeks": int, "months": int, "pid": int, "day": int})#.iloc[:5000]

    X_train, X_test, Y_train = application_train, application_test, target_train.loc[application_train.index]#.iloc[:5000]

    # Concat for speed and lag retention
    X_test = pd.concat((X_train,X_test),axis=0)

    # Add lags
    for var in options['var_to_lag']: 
        # X_train =  add_lags(X_train,range(options['lag1'],options['lag2']+1),options['lag_step'],var)
        X_test  =  add_lags(X_test,range(options['lag1'],options['lag2']+1),options['lag_step'],var)

    # Add cross
    if options['Cross']:
        add_inter = lambda x: x['NLV'] * x['LS']
        # X_train['NLV_LS'] = add_inter(X_train)
        X_test['NLV_LS']  = add_inter(X_test)

    # Separate Test and Train sets
    X_train = X_test.loc[application_train.index]
    X_test  = X_test.loc[application_test.index]

    # OLS-Residuals
    y_train_res, model_ols = stats.get_resid_ols(X_train.drop('pid',axis=1), Y_train.values.flatten())
    y_test_res = model_ols.predict(X_test.drop('pid',axis=1))

    # Drop day
    X_train.drop('day',axis=1,inplace=True)
    X_test.drop('day',axis=1,inplace=True)
    
    # Set types
    cat = X_train.select_dtypes(include='int').columns

    X_train[cat] = X_train[cat].astype('category')
    X_test[cat]  = X_test[cat].astype('category')

    print('Categorical variables are', ', '.join(X_train.select_dtypes(include='category').columns),'\b.')

    # Check series
    assert (X_train.columns == X_test.columns).all(), "Train and test sets have different columns."+', '.join(X_train.columns[~(X_train.columns == X_test.columns)])
    # Check data types
    assert (X_train.dtypes == X_test.dtypes).all(), "Train and test sets have different data types: "+', '.join(X_train.columns[~(X_train.dtypes == X_test.dtypes)])
    # Check index 
    assert (X_train.index == Y_train.index).all(), "Train and test index not equal."
    # Check lags
    assert (X_train.query('pid==1')[options['var_to_lag'][0]][:-1].values ==\
    X_train.query('pid==1')[options['var_to_lag'][0]+'_L'+str(options['lag1'])][options['lag1']:].values).all(), "Lags misadjusted."


    # Get Dataset
    X_idx         = range(0,X_train.shape[0])
    train_dataset = Dataset(X_train, label=y_train_res, free_raw_data=False, )
        
    if options["save_inputs"]:
        import os
        if not os.path.exists("last_run"):
            os.makedirs("last_run")
        save_pkl("train_dataset",train_dataset) 
        save_pkl("y_test_res",y_test_res) 
        save_pkl("X_test",X_test) 
        save_pkl("X_idx",X_idx)
        save_pkl("model_ols",model_ols)
        save_pkl("options",options)
        
    return train_dataset, y_test_res, X_test, X_idx

def save_pkl(fname,obj):
    import pickle
    print('Saving', (fname),'\b.')
    with open('last_run\\'+(fname)+'.pkl', 'wb') as file:
        pickle.dump(obj, file)
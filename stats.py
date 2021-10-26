import numpy as np
import sklearn.linear_model as linear_model

def get_resid_ols(X, Y):
    model = linear_model.LinearRegression().fit(X,Y)
    # model = linear_model.SGDRegressor(alpha=0).fit(X,Y)
    res   = Y - model.predict(X)
    print('OLS RSS is ', sum(res**2)/len(res),'\b.')
    return res, model

def learning_rate_010_decay_power_099(current_iter):
    base_learning_rate = 0.1
    lr = base_learning_rate  * np.power(.99, current_iter)
    return lr if lr > 0.005 else 0.005

def learning_rate_010_decay_power_0995(current_iter):
    base_learning_rate = 0.1
    lr = base_learning_rate  * np.power(.999, current_iter)
    return lr if lr > 0.005 else 0.005

def learning_rate_005_decay_power_099(current_iter):
    base_learning_rate = 0.05
    lr = base_learning_rate  * np.power(.99, current_iter)
    return lr if lr > 0.005 else 0.005
        
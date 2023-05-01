# for presentation purposes
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# visualize 
import matplotlib.pyplot as plt

import seaborn as sns

import wrangle as wr

# working with dates
from datetime import datetime

# to evaluated performance using rmse
from sklearn.metrics import mean_squared_error
from math import sqrt 

# for tsa 
import statsmodels.api as sm

# holt's linear trend model. 
from statsmodels.tsa.api import Holt, ExponentialSmoothing

from datetime import timedelta, datetime
from statsmodels.tsa.arima.model import ARIMA

# Plt defaults
# plt.style.use('seaborn-whitegrid')
plt.rc('figure', figsize=(14, 8))
plt.rc('font', size=16)
import seaborn as sns; sns.color_palette("tab10")



def model():
    df = wr.wrangle()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    df_resampled = df.resample('d')[['Quantity','Revenue']].sum()
    train_size = int(round(df_resampled.shape[0] * 0.5))
    validate_size = int(round(df_resampled.shape[0] * 0.3))
    test_size = int(round(df_resampled.shape[0] * 0.2))
    train = df_resampled[:train_size]
    validate_end_index = train_size + validate_size
    validate = df_resampled[train_size:validate_end_index]
    test = df_resampled[validate_end_index:]
    return train, validate, test

train, validate, test = model()
last_revenue = train['Revenue'][-1:][0]
last_quantity = train['Quantity'][-1:][0]
eval_df = pd.DataFrame(columns=['model_type', 'target_var', 'rmse'])

yhat_df = pd.DataFrame(
    {'Revenue': [last_revenue],
     'Quantity': [last_quantity]},
    index=validate.index)

def evaluate(target_var):
    '''
    This function will take the actual values of the target_var from validate, 
    and the predicted values stored in yhat_df, 
    and compute the rmse, rounding to 0 decimal places. 
    it will return the rmse. 
    '''
    rmse = round(sqrt(mean_squared_error(validate[target_var], yhat_df[target_var])), 0)
    return rmse


def plot_and_eval(target_var):
    '''
    This function takes in the target var name (string), and returns a plot
    of the values of train for that variable, validate, and the predicted values from yhat_df. 
    it will als lable the rmse. 
    '''
    plt.figure(figsize = (12,4))
    plt.plot(train[target_var], label='Train', linewidth=1)
    plt.plot(validate[target_var], label='Validate', linewidth=1)
    plt.plot(yhat_df[target_var])
    plt.title(target_var)
    rmse = evaluate(target_var)
    print(target_var, '-- RMSE: {:.0f}'.format(rmse))
    plt.show()



def append_eval_df(model_type, target_var):
    '''
    this function takes in as arguments the type of model run, and the name of the target variable. 
    It returns the eval_df with the rmse appended to it for that model and target_var. 
    '''
    rmse = evaluate(target_var)
    d = {'model_type': [model_type], 'target_var': [target_var],
        'rmse': [rmse]}
    d = pd.DataFrame(d)
    return eval_df.append(d, ignore_index = True)

def final_plot(target_var):
    plt.figure(figsize=(12,4))
    plt.plot(train[target_var], label='train')
    plt.plot(validate[target_var], lavzbel='validate')
    plt.plot(test[target_var], label='test')
    plt.plot(yhat_df[target_var], alpha=.5, label='forecast')
    plt.title(target_var)
    plt.legend()
    plt.show()

def make_predictions(sales=None, quantity=None):
    yhat_df = pd.DataFrame({'Revenue': [sales],
                           'Quantity': [quantity]},
                          index=validate.index)
    return yhat_df

yhat_df = make_predictions()

last_revenue = train['Revenue'][-1:][0]
last_quantity = train['Quantity'][-1:][0]


yhat_df = pd.DataFrame(
    {'Revenue': [last_revenue],
     'Quantity': [last_quantity]},
    index=validate.index)

def final_model():
    col = 'Revenue'
    # create the Holt object 
    model = Holt(train[col], exponential=False, damped=True)
    # fit the model 
    model = model.fit(optimized=True)
    # make predictions for each date in validate 
    yhat_items = model.predict(start = validate.index[0],
                           end = validate.index[-1])
    # add predictions to yhat_df
    yhat_df[col] = round(yhat_items, 2)
    yhat_df = test + train.diff().mean()

    yhat_df.index = test.index + pd.Timedelta('17W')
    for col in train.columns: 
        final_plot(col)


def yhat():
    last_revenue = train['Revenue'][-1:][0]
    last_quantity = train['Quantity'][-1:][0]


    yhat_df = pd.DataFrame(
      {'Revenue': [last_revenue],
       'Quantity': [last_quantity]},
      index=validate.index)

    return yhat_df

















































from pprint import pprint
from pandas import datetime
from matplotlib import pyplot
import pandas as pd
import numpy as np
import time
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
# pylint: disable=maybe-no-member

###
### AR: Autoregression. A model that uses the dependent relationship between an observation and some number of lagged observations.
### I: Integrated. The use of differencing of raw observations (e.g. subtracting an observation from an observation at the previous time step) in order to make the time df stationary.
### MA: Moving Average. A model that uses the dependency between an observation and a residual error from a moving average model applied to lagged observations.
###


def arima (series,period):
    
    # downsample data by hour (take max )
    # Too many records for Hourly so weekly
    df = series.resample('D').max()
    #df = df.resample('W').mean()

    # replace nan values by linear interpolation
    df = df.interpolate(method='time')
    df = df.fillna(0)
    #autocorrelation_plot(df)
    autocorrelation_plot(df)
    pyplot.show()

    # fit model
    # First, we fit an ARIMA(5,1,0) model. 
    # This sets the lag value to 5 for autoregression, 
    # uses a difference order of 1 to make the time df stationary, 
    # and uses a moving average model of 0.
    model = ARIMA(df, order=(5,1,0))
    model_fit = model.fit(disp=0)
    print(model_fit.summary())
    # plot residual errors
    residuals = pd.DataFrame(model_fit.resid)
    residuals.plot()
    pyplot.show()
    residuals.plot(kind='kde')
    pyplot.show()
    print(residuals.describe())

    # USE ARIMA
    X = df.values
    train, test = X[:-period], X[-period:]
    history = [x for x in train]
    predictions = list()

    for t in range(len(test)):
	    model = ARIMA(history, order=(5,1,0))
	    model_fit = model.fit(disp=0)
	    output = model_fit.forecast()
	    yhat = output[0]
	    predictions.append(yhat)
	    obs = test[t]
	    history.append(obs)
	    print('predicted=%f, expected=%f' % (yhat, obs))
    error = mean_squared_error(test, predictions)
    print('Test MSE: %.3f' % error)
    # plot
    pyplot.plot(test)
    pyplot.plot(predictions, color='red')
    pyplot.show()
    

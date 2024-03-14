
#importing necessary libraries
import numpy as np 
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import numpy as np 
import pandas as pd 
from pandas import DataFrame
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
import statsmodels.tsa.api as smt
import statsmodels.formula.api as smf

from sklearn.metrics import mean_squared_error

%matplotlib inline


#File path
import os
if os.path.isfile("portfolio_data.csv"):
    print("File exists.")
else:
    print("File does not exist.")
#Reading file
with open("portfolio_data.csv", "r") as file:
    for i in range(5):  
        line = file.readline()
        print(line)
#Reading Dataset
df = pd.read_csv('portfolio_data.csv', parse_dates=['Date']) #File name,col name
df.head(3)
#Describing dataset
print (df.describe())
print ("=============================================================")
print (df.dtypes)
df_ts = df1.set_index('Date')
df_ts.sort_index(inplace=True)
print (type(df_ts))
print (df_ts.head(3))
print ("========================")
print (df_ts.tail(3))
 #Plotting
df_ts.plot()

def test_stationarity(timeseries):
    # Perform Dickey-Fuller test:
    from statsmodels.tsa.stattools import adfuller
    print('Results of Dickey-Fuller Test:')
    print ("==============================================")

#Stationarity test
    dftest = adfuller(timeseries, autolag='AIC')
    
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#lags Used', 'Number of Observations Used'])
    
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    
    print(dfoutput)
ts = df_ts['AMZN']
test_stationarity(ts)

ts = df_ts['DPZ']
test_stationarity(ts)

ts = df_ts['BTC']
test_stationarity(ts)

ts = df_ts['NFLX']
test_stationarity(ts)

#Visualization
rolmean = ts.rolling(window=12).mean()
rolvar = ts.rolling(window=12).std()

plt.plot(ts, label='Original')
plt.plot(rolmean, label='Rolling Mean')
plt.plot(rolvar, label='Rolling Standard Variance')
plt.legend(loc='best')
plt.title('Rolling Mean & Standard Deviation')
plt.show(block=False)





import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import statsmodels.api as sm
import numpy as np
import scipy.stats as scs
from datetime import datetime
from statsmodels.tsa.arima_model import ARIMA
from dateutil.relativedelta import relativedelta
import warnings
import itertools
plt.style.use('fivethirtyeight')

driver = "{SQL Server}"
server = "DESKTOP-HLCA1Q9"
database = "forecast_ex"
trusted = "yes"

try:
	conn = pyodbc.connect("Driver="+driver+";"
                      "Server="+server+";" 
                      "Database="+database+";" 
                      "Trusted_Connection="+trusted+";")

	df = pd.read_sql('SELECT * FROM forecast_ex.dbo.MonthWiseSalesSum', conn)
	print("Connection established and Data read from database")
except:
	print("Data failed read from database")

dates = df['Month']
sales = df['Sales']
months_forecast = 12

train = pd.DataFrame({'Month': dates, 'Sales': sales})
train["Sales"] = train["Sales"].astype("float64")
train["Month"] = pd.to_datetime(train["Month"])
train = train.set_index(["Month"])
train.index = pd.DatetimeIndex(train.index.values, freq=train.index.inferred_freq)
train['Dates'] = train.index

#############################################################################################################
train2 = pd.DataFrame({'Month': train["Dates"], 'Sales': train["Sales"]})
train2["Sales"] = train2["Sales"].astype("float64")
train2["Month"] = pd.to_datetime(train2["Month"])
train2 = train2.set_index(["Month"])
train2.index = pd.DatetimeIndex(train2.index.values, freq=train2.index.inferred_freq)

train2.head()

# FIGURING OUT BEST PARAMETER VALUES FOR MODEL

# Define the p, d and q parameters to take any value between 0 and 2
p = d = q = range(0, 2)

# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))

AIC = []
parm_ = []
for param in pdq:
    try:
        mod = ARIMA(train2.Sales, order=param)

        results = mod.fit()
        AIC.append(results.aic)
        parm_.append(param)
        #print('ARIMA{} - AIC:{}'.format(param, results.aic))
    except:
        print("ERROR")

pos = AIC.index(min(AIC))
order = parm_[pos]
#############################################################################################################

# (1, 0, 0)
# (1, 0, 1)


mod = ARIMA(train.Sales, order=(1, 0, 0))
results = mod.fit()

start = pd.datetime.strptime("2019-08-01", "%Y-%m-%d")
date_list = [start + relativedelta(months=x) for x in range(0,months_forecast)]
future = pd.DataFrame(index=date_list, columns= train.columns)
train = pd.concat([train, future])

train['forecast'] = results.predict(start = len(df), end = len(train), dynamic= True)  
train[['Sales', 'forecast']].plot(figsize=(12, 5), kind='line')
plt.ylabel("Sales")
plt.xlabel("Years")
plt.title("Forecasted Sales 2019-20")

dates = train.index
sales = train['Sales']
future = train['forecast']
out_df = pd.DataFrame({'Month': dates, 'Sales': sales, 'Forecast': future})
out_df["Sales"] = out_df["Sales"].astype("float64")
out_df["Forecast"] = out_df["Forecast"].astype("float64")
out_df["Month"] = pd.to_datetime(out_df["Month"])
out_df.reset_index(drop=True,inplace=True)

data1 = out_df["Month"]
data2 = out_df["Sales"]
data3 = out_df["Forecast"]

engine = create_engine("mssql+pyodbc://"+server+"/"+database+"?driver="+driver)

out_df.to_sql(name='MonthWiseSalesSum_new',con=engine,if_exists='replace')

print("Output Exported")
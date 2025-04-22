import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.metrics import r2_score,mean_squared_error
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import seasonal_decompose,ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor





plt.style.use('fivethirtyeight')


"""Using IBM"""

IBM = yf.download(tickers="IBM",start="1990-01-01",end="2025-04-20")['Close']
IBM = IBM.reset_index()


df = IBM[['Date','IBM']]


df['Date'] = pd.to_datetime(df['Date'])

df = df.set_index('Date')





df.isnull().sum()
df.duplicated().sum()
df.drop_duplicates(inplace=True)


plt.figure(figsize=(10,6))
plt.plot(df)
plt.xlabel('Date')
plt.ylabel('Closing Price For IBM')
plt.show()

"""Copying 'df' to show the index for each time period"""


df1 = df.copy()

def create_values(df1):
    df1 = df1.copy()
    df1['year'] = df1.index.year
    df1['month'] = df1.index.month
    df1['dayofyear'] = df1.index.dayofyear
    df1['day'] = df1.index.day
    df1['quarter'] = df1.index.quarter

    return df1
    
df1 = create_values(df1)
    



plt.figure(figsize=(10,6))
sns.lineplot(x='year',y='IBM',hue='quarter',data=df1)
plt.xlabel('Year')
plt.ylabel('Closing Price')
plt.title("Closing Price by year and quarter")
plt.legend()
plt.show()

plt.figure(figsize=(10,6))
sns.lineplot(x='year',y='IBM',hue='day',data=df1)
plt.title("CLosing Price by each year by day")
plt.xlabel("Year")
plt.ylabel("IBM Closing Price")
plt.legend()
plt.show()

plt.figure(figsize=(10,6))
sns.pointplot(x='month',y='IBM',hue='quarter',data=df1)
plt.xlabel("Month")
plt.ylabel("IBM Closing Price")
plt.title("IBM CLosing Price by Month and Quarter")
plt.legend()
plt.show()






decomp = seasonal_decompose(df['IBM'],model="additive",period=30)
decomp.plot().show()




def test_stationary(timeseries):
    
    movingaverage = timeseries.rolling(window=30).mean()
    movingstd = timeseries.rolling(window=30).std()
    plt.plot(timeseries,color="blue",label="original")
    plt.plot(movingaverage,color="green",label="Moving Average")
    plt.plot(movingstd,color="black",label="Moving Standard Deviation")
    plt.title("TimeSeries For IBM Closing Prices")
    plt.legend(loc="best")
    plt.title("Rolling Mean and Rolling Standard Deviation")
    plt.show(block=False)
    
    print('Results from Dickey Fueller\n')
    df_test = adfuller(timeseries['IBM'],autolag="AIC")
    dfoutput = pd.Series(df_test[0:4],index=['Test Statistic','p-value',"'#Lags Used","Number of Observations Used"])
    for key,value in df_test[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
    
    
test_stationary(df)

""" df_log first"""


df_log = np.log(df)

plt.figure(figsize=(10,6))
plt.plot(df,label="original")
plt.plot(df_log,label="Original With No Variance")
plt.legend()
plt.show()

"""testing stationary of df_log"""

df_log.dropna(inplace=True)
test_stationary(df_log)




""" Need the p-value to be lower to be stationary First-order Differncing"""
df_diff = df.diff(periods=1)

plt.plot(figsize=(10,6))
plt.plot(df_diff)
plt.xlabel("Date")
plt.ylabel('Closing Prices With First-Order Differncing')
plt.show()


df_diff.dropna(inplace=True)

test_stationary(df_diff)



"""Ok, I made a mistake because I did this in a hour this morning(look and the end date),
I need to test more if this truly is stationary"""

df_diff2 = df_diff.diff(periods=1)


plt.figure(figsize=(10,6))
plt.plot(df,label="original")
plt.plot(df_diff,label="First Order Diff")
plt.plot(df_diff2,label="second order diff")
plt.legend()
plt.show()



df_diff2.dropna(inplace=True)
test_stationary(df_diff2)


""" ok, a couple of more tests to make sure"""

df_log_diff = np.log(df_diff)

plt.figure(figsize=(10,6))
plt.plot(df,label="original")
plt.plot(df_diff,label="First Order Diff")
plt.plot(df_diff2,label="second order diff")
plt.plot(df_log_diff,label="1st Order Log Diff")
plt.legend()
plt.show()

df_log_diff.dropna(inplace=True)
test_stationary(df_log_diff)

df_new = df_log_diff - df_diff

plt.plot(figsize=(10,6))
plt.plot(df,label="original")
plt.plot(df_diff,label="first order")
plt.plot(df_diff2,label="Second Order")
plt.plot(df_log_diff,label="First Order Log")
plt.plot(df_new,label="new")
plt.legend()
plt.show()

df_new.dropna(inplace=True)
test_stationary(df_new)

df_diff_rollingmean = df_diff.rolling(window=30).mean()



plt.plot(figsize=(10,6))
plt.plot(df,label="original")
plt.plot(df_diff,label="first order")
plt.plot(df_diff2,label="Second Order")
plt.plot(df_log_diff,label="First Order Log")
plt.plot(df_new,label="new")
plt.plot(df_diff_rollingmean,label="Rolling Mean of Log First Order Diff")
plt.legend()
plt.show()





""" moving seasonal decomp with differncing"""


decomp = seasonal_decompose(df_diff,model="additive",period=15)

trend = decomp.trend
seasonal = decomp.seasonal
residual = decomp.resid

plt.subplot(211)
plt.plot(df_diff,label="First-Order Differencing")
plt.legend(loc="best")
plt.show()

plt.subplot(212)
plt.plot(trend,label="Trend")
plt.legend(loc="best")
plt.show()



df_decomp = residual
residual.dropna(inplace=True)


decomp_mean = df_decomp.rolling(window=30).mean()
decomp_std = df_decomp.rolling(window=30).std()

plt.figure(figsize=(10,6))
plt.plot(df_decomp,label="Orginal Decomposed")
plt.plot(decomp_mean,label="Rolling Mean Decomp")
plt.plot(decomp_std,label="Rolling STD Decomp")
plt.legend()
plt.show()



model1 = ARIMA(df,order=(1,2,0)).fit()
plt.plot(df,color="blue")
plt.plot(model1.fittedvalues, color='red')
plt.show()


model2 = ARIMA(df,order=(1,2,0)).fit()
plt.figure(figsize=(10,6))
plt.plot(df_diff,color="black")
plt.plot(model2.fittedvalues,color="red")
plt.show()




"""Arima is outdated. Break out the neural networks and regular ML models"""


""" Forecast Using XGBboost and other regular ML models first"""

df = create_values(df)
df.head(10)


X = df.drop('IBM',axis=1)
y= df['IBM']



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=42)


scaler = MinMaxScaler(feature_range=(0,1))

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


""" A user-Defined function for metrics"""

def evaluate(y_test,pred,model_name,r2,mse,cv_scores):


    result = {
        "Model": model_name,
        "R2": r2,
        "MSE": mse,
        "Cross-val Scores":cv_scores.mean()
    }

    return result


model_dict = []
models = {
    "LinearRegression":LinearRegression(),
    "xgboost":XGBRegressor(),
    "gradientboostingregressor":GradientBoostingRegressor()
}


for model_name,model in models.items():
    model.fit(X_train_scaled,y_train)
    pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test,pred)
    r2 = r2_score(y_test,pred)
    cv_scores = cross_val_score(model, X_train_scaled,y_train,cv=10,scoring="neg_mean_squared_error")
    model_results = evaluate(y_test, pred, model_name,r2,mse,cv_scores)
    model_dict.append(model_results)



df_results = pd.DataFrame(model_dict)
print(df_results.head())

xgb = XGBRegressor().fit(X_train_scaled,y_train)
xgb_predictions = xgb.predict(X_test_scaled)
print(f'Predictions using the Best Model: {xgb_predictions}')
print(f'Actual Values: {y_test}')



pred_vs_actual = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': xgb_predictions
}, index=y_test.index)



pred_vs_actual.sort_index(inplace=True)
print('Predicted Vs Actual Prices\n')
print(pred_vs_actual)


plt.figure(figsize=(14,7))
plt.plot(pred_vs_actual.index, pred_vs_actual['Actual'], label='Actual', color='blue')
plt.plot(pred_vs_actual.index, pred_vs_actual['Predicted'], label='Predicted', color='red')
plt.title('IBM Closing Price: Actual vs Predicted')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.tight_layout()
plt.show()



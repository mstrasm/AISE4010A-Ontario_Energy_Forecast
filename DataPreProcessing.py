import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import zscore
from sklearn.preprocessing import  StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose

df = pd.read_csv("ontario_electricity_demand.csv")
date = []
for i in range(len(df)):
    date_str = df['date'].iloc[i]
    hour_str = (df['hour'].iloc[i])-1
    datetime_obj = datetime.strptime(f"{date_str} {hour_str}:00", "%Y-%m-%d %H:%M")
    date.append(datetime_obj)

date = pd.DataFrame(date)
df = pd.concat([date,df],axis=1)
df = df.drop(columns=['date','hour'])
df.columns = ['date_time', 'hourly_demand', 'hourly_average_price']
df = df.drop(df.index[181200:])
df = df.iloc[5880:]
df = df.set_index("date_time")

import matplotlib.pyplot as plt

# Select a specific day to plot (e.g., '2019-01-01')
specific_day = '2019-01-01'
end_date = '2019-01-07'


# Filter the data to include only the selected day
week_data = df.loc[specific_day:end_date]

# Plot hourly demand for the selected day
plt.figure(figsize=(12, 6))
plt.plot(week_data.index, week_data['hourly_demand'], label=f'Hourly Demand on {specific_day}')
plt.title(f'Ontario Electricity Hourly Demand on {specific_day}')
plt.xlabel('Date Time')
plt.ylabel('Hourly Demand')
plt.xticks(rotation=45)  # Rotate x-axis labels for readability
plt.legend()
plt.tight_layout()
plt.show()



# Decompose for daily seasonality
daily_decompose = seasonal_decompose(df['hourly_demand'], model='additive', period=8760)

# Plot results
daily_decompose.plot()
plt.show()

def create_features(df):
    
    df['hour'] = df.index.hour
    df['dayofweek']=df.index.dayofweek
    df['quarter']=df.index.quarter
    df['month']=df.index.month
    df['year']=df.index.year
    df['dayofyear']=df.index.dayofyear
    
    return df

df = create_features(df)

#80% training, 20% testing
train = df.loc[df.index < '2019-01-01 00:00:00']
test = df.loc[df.index >= '2019-01-01 00:00:00']

# Standardize features and target
scaler_X = StandardScaler()
scaler_y = StandardScaler()
import pandas as pd

# Create a lag feature for hourly_demand

train['hourly_demand_lag_1'] = train['hourly_demand'].shift(1)  # Lag by 1 hour
train['hourly_demand_lag_24'] = train['hourly_demand'].shift(24)  # Lag by 24 hours
test['hourly_demand_lag_1'] = test['hourly_demand'].shift(1)  # Lag by 1 hour
test['hourly_demand_lag_24'] = test['hourly_demand'].shift(24)  # Lag by 24 hours

train = train.dropna()
test = test.dropna()

X_train = scaler_X.fit_transform(train.drop(['hourly_demand', "hourly_average_price"], axis=1))
y_train = scaler_y.fit_transform(train[['hourly_demand']])
X_test = scaler_X.fit_transform(test.drop(['hourly_demand','hourly_average_price'], axis=1))
y_test = scaler_y.fit_transform(test[['hourly_demand']])

WINDOW_SIZE = 336

def create_sliding_window(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size, :-1])  # Features
        y.append(data[i + window_size, -1])    # Target: hourly_demand
    return np.array(X), np.array(y)

# Create sliding windows
X_train_sliding, y_train_sliding = create_sliding_window(np.hstack([X_train, y_train]), WINDOW_SIZE)
X_test_sliding, y_test_sliding = create_sliding_window(np.hstack([X_test, y_test]), WINDOW_SIZE)

print(X_train_sliding.shape)


#print(df.loc['2006-1-1 00:00:00'])

# Extract test dates for plotting
test_dates = test.index[WINDOW_SIZE:]  # Align with the sliding window output

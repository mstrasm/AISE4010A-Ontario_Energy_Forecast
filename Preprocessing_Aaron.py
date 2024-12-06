import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("C:\\Users\\sohra\\School\\Year-4\\AISE4010\\proj\\Resources\\ontario_electricity_demand.csv")

# Combine date and hour into a datetime object
df['date_time'] = pd.to_datetime(df['date'] + ' ' + (df['hour'] - 1).astype(str) + ':00', format='%Y-%m-%d %H:%M')
df.drop(columns=['date', 'hour'], inplace=True)


df.rename(columns={'hourly_demand': 'demand', 'hourly_average_price': 'price'}, inplace=True)


df.set_index('date_time', inplace=True)


df = df[(np.abs(df['price'] - df['price'].mean()) / df['price'].std()) <= 3]

# Add temporal features
df['hour'] = df.index.hour
df['dayofweek'] = df.index.dayofweek
df['month'] = df.index.month
df['year'] = df.index.year
df['dayofyear'] = df.index.dayofyear


def add_lag_features(data, target_col, lags):
    for lag in range(1, lags + 1):
        data[f'{target_col}_lag_{lag}'] = data[target_col].shift(lag)
    return data


lags = 3  
df = add_lag_features(df, 'demand', lags)

# Drop rows with NaN values due to lag features
df.dropna(inplace=True)

# Scale numerical features
scaler = StandardScaler()
numerical_features = ['demand', 'price', 'hour', 'dayofyear'] + [f'demand_lag_{i}' for i in range(1, lags + 1)]
df[numerical_features] = scaler.fit_transform(df[numerical_features])


train, test = train_test_split(df, test_size=0.2, shuffle=False)

# Verify the data
print("Training Data:")
print(train.head())
print("\nTesting Data:")
print(test.head())


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, LSTM, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import  StandardScaler
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from DataPreProcessing import X_train_sliding, y_train_sliding,X_test_sliding, y_test_sliding, scaler_x, scaler_y
import pandas as pd

WINDOW_SIZE = 336 
EPOCHS = 50
BATCH_SIZE = 32 
LEARNING_RATE = 0.001 
DROPOUT_RATE = 0.5
FILTERS = 96
KERNEL_SIZE = 7
LSTM_UNITS = 50


def build_model(input_shape):
    model = Sequential()
    model.add(Conv1D(filters=FILTERS, kernel_size=KERNEL_SIZE, activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(LSTM(units=LSTM_UNITS, activation="tanh", return_sequences=False)) 
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(units=1)) 
    optimizer = Adam(learning_rate=LEARNING_RATE, clipvalue=1.0) 
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

print(X_train_sliding.shape[1], X_train_sliding.shape[2])

input_shape = (X_train_sliding.shape[1], X_train_sliding.shape[2])
model = build_model(input_shape)

model.summary()
history = model.fit(
    X_train_sliding, y_train_sliding,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    verbose=1
)

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

y_pred_scaled = model.predict(X_test_sliding)

y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_original = scaler_y.inverse_transform(y_test_sliding.reshape(-1, 1))

plt.figure(figsize=(10, 6))
plt.plot(y_test_original, label='Actual hourly_demand', marker='o')
plt.plot(y_pred, label='Forecasted hourly_demand', linestyle='--')
plt.title('Actual vs Forecasted Values (hourly_demand)')
plt.xlabel('Time Steps')
plt.ylabel('Hourly Demand')
plt.legend()
plt.grid(True)
plt.show()

mae_demand = mean_absolute_error(y_test_original, y_pred)
print(f'Mean Absolute Error (hourly_demand): {mae_demand:.2f}')

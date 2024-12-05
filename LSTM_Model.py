from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import pandas as pd  # For handling dates
from keras.callbacks import EarlyStopping
from dataPreProc import X_train_sliding, y_train_sliding, X_test_sliding, y_test_sliding, scaler_X, scaler_y, test_dates

# Ensure test_dates is a list or array of datetime objects corresponding to y_test_sliding.

def buildLSTM():
    model = Sequential()
    model.add(LSTM(20, return_sequences=False, input_shape=(X_train_sliding.shape[1], X_train_sliding.shape[2])))
    model.add(Dropout(rate=0.3))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

model = buildLSTM()

history = model.fit(
    X_train_sliding, y_train_sliding,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    verbose=1,
    callbacks=[EarlyStopping(monitor='val_loss', patience=3)]
)

# Plot training vs validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Make predictions
y_pred_scaled = model.predict(X_test_sliding)

# Reshape predictions and test targets to 2D
y_pred_scaled = y_pred_scaled.reshape(-1, 1)
y_test_sliding = y_test_sliding.reshape(-1, 1)

# Inverse transform predictions and test targets
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_original = scaler_y.inverse_transform(y_test_sliding)

# Ensure test_dates is aligned with y_test_original
test_dates = pd.to_datetime(test_dates)  # Convert to datetime if necessary

# Plot actual vs forecasted values with dates
# Plot actual vs forecasted values with dates
plt.figure(figsize=(10, 6))
plt.plot(test_dates, y_test_original, label='Actual hourly_demand', marker='o')
plt.plot(test_dates, y_pred, label='Forecasted hourly_demand', linestyle='--')
plt.title('Actual vs Forecasted Values (hourly_demand)')
plt.xlabel('Dates')
plt.ylabel('Hourly Demand')
plt.legend()
plt.grid(True)
plt.gcf().autofmt_xdate()  # Rotate date labels for better readability
plt.show()


# Evaluate performance
mae_demand = mean_absolute_error(y_test_original, y_pred)
print(f'Mean Absolute Error (hourly_demand): {mae_demand:.2f}')

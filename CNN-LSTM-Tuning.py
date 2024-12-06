import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, LSTM, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from keras_tuner import RandomSearch
from FinalProject.DataPreProcessing import X_train_sliding, y_train_sliding,X_test_sliding, y_test_sliding, scaler_X, scaler_y


WINDOW_SIZE = 336  
BATCH_SIZE = 32  
EPOCHS = 10  

scaler_X = StandardScaler()
scaler_y = StandardScaler()

def build_model(hp):
    model = Sequential()
    model.add(Conv1D(
        filters=hp.Int('filters', min_value=32, max_value=128, step=32),
        kernel_size=hp.Choice('kernel_size', values=[3, 5, 7]),
        activation='relu',
        input_shape=(WINDOW_SIZE, X_train_sliding.shape[2])
    ))
    model.add(BatchNormalization())
    model.add(LSTM(
        units=hp.Int('lstm_units', min_value=10, max_value=50, step=10),
        activation="tanh",
        return_sequences=False
    ))

    model.add(Dropout(rate=hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(Dense(units=1))
    model.compile(
        optimizer=Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
        loss='mse',
        metrics=['mae']
    )
    return model

tuner = RandomSearch(
    build_model,
    objective='val_loss',  
    max_trials=10,        
    executions_per_trial=1,  
    directory='my_tuning_results',
    project_name='cnn_lstm_tuning'
)

tuner.search(
    X_train_sliding, y_train_sliding,
    epochs=10,  
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    verbose=1
)

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The optimal number of filters is {best_hps.get('filters')}.
The optimal kernel size is {best_hps.get('kernel_size')}.
The optimal number of LSTM units is {best_hps.get('lstm_units')}.
The optimal dropout rate is {best_hps.get('dropout_rate')}.
The optimal learning rate is {best_hps.get('learning_rate')}.
""")

best_model = tuner.hypermodel.build(best_hps)
history = best_model.fit(
    X_train_sliding, y_train_sliding,
    epochs=20,
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

y_pred_scaled = best_model.predict(X_test_sliding)

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

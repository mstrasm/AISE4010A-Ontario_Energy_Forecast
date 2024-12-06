import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import Huber
import matplotlib.pyplot as plt
from Preprocessing import df, train, test  # Import the preprocessed dataframes

# Enable memory growth for the GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth enabled for GPU.")
    except RuntimeError as e:
        print(e)

# Verify GPU availability
if tf.config.list_physical_devices('GPU'):
    print("GPU is available!")
else:
    print("GPU is NOT available. Check your TensorFlow installation.")

# Sliding window function
def create_sliding_windows(data, target_col, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data.iloc[i:i + window_size].drop(columns=[target_col]).values)
        y.append(data.iloc[i + window_size][target_col])
    return np.array(X), np.array(y)


window_size = 24 
X_train, y_train = create_sliding_windows(train, 'demand', window_size)
X_test, y_test = create_sliding_windows(test, 'demand', window_size)


input_shape = (X_train.shape[1], X_train.shape[2])

optimizer = Adam(learning_rate=0.001, clipvalue=1.0)

model = Sequential([
    GRU(64, activation='relu', return_sequences=False, input_shape=input_shape),
    Dropout(0.2),
    
    Dense(1)
])

model.compile(optimizer=optimizer, loss=Huber(), metrics=['mae'])
early_stopping = EarlyStopping(monitor='loss', patience=2, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=16,
    batch_size=128,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1)
# Evaluate the model
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}")



# Plot all training and validation loss points
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')  
plt.plot(history.history['val_loss'], label='Validation Loss')  
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True) 
plt.show()


# Predict and plot results
predictions = model.predict(X_test).flatten()

plt.figure(figsize=(10, 6))
plt.plot(y_test, label="Actual", alpha=0.7)  
plt.plot(predictions, label="Predicted", alpha=0.7)
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Actual vs Predicted')
plt.legend()
plt.show()
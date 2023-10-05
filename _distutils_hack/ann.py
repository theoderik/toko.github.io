import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# Unduh data historis harga saham UNVR.JK dari Yahoo Finance
stock_data = yf.download('GOTO.JK', start='2022-09-27', end='2023-09-27')

# Ambil kolom harga penutupan (Close) sebagai target prediksi
closing_prices = stock_data['Close'].values.reshape(-1, 1)

# Normalisasi data harga saham
scaler = MinMaxScaler()
scaled_closing_prices = scaler.fit_transform(closing_prices)

# Membagi data menjadi data pelatihan dan pengujian
train_size = int(len(scaled_closing_prices) * 0.8)
train_data = scaled_closing_prices[:train_size]
test_data = scaled_closing_prices[train_size:]

# Menyiapkan data untuk model
def prepare_data(data, time_steps):
  X, y = [], []
  for i in range(len(data) - time_steps):
    X.append(data[i:i+time_steps])
    y.append(data[i+time_steps])
    return np.array(X), np.array(y)

time_steps = 30 # Jumlah data sebelumnya yang akan digunakan
X_train, y_train = prepare_data(train_data, time_steps)
X_test, y_test = prepare_data(test_data, time_steps)

# Membangun model ANN
model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=time_steps))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=1, activation='linear'))

# Kompilasi model
model.compile(optimizer='adam', loss='mean_squared_error',
              metrics=['mean_absolute_error'])

# Pelatihan model
history = model.fit(X_train, y_train, epochs=50, batch_size=32,
                    validation_data=(X_test, y_test))

# Evaluasi model
loss, mae = model.evaluate(X_test, y_test)
print(f"Loss: {loss:.4f}, Mean Absolute Error: {mae:.4f}")

# Visualisasi hasil pelatihan
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
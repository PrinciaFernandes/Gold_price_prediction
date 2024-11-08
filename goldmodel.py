import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


# Load your dataset
df1 = pd.read_csv('C:/Users/Princia/colab/Gold  Historical Data.csv')
df2 = pd.read_csv('C:/Users/Princia/colab/Gold Historical Data 2.csv')
df3 = pd.read_csv("C:/Users/Princia/colab/Gold Futures Historical Data4.csv")

df1 = pd.concat([df1,df2, df3]).reset_index(drop = True)
df1 = df1.drop(['Vol.','Change %'],axis=1)

numcols=['Price','Open','High','Low']
df1[numcols]=df1[numcols].replace({',':''},regex=True)
df1[numcols]=df1[numcols].astype('float64')
df1['Date'] = pd.to_datetime(df1['Date'],format='%d/%m/%Y')
df  = df1[df1['Price'] > 900]
df.dropna(inplace = True)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df['Price'].values.reshape(-1, 1))

# Create sequences
def create_sequences(data, window_size):
    sequences = []
    labels = []
    for i in range(len(data) - window_size):
        sequences.append(data[i:i + window_size])
        labels.append(data[i + window_size])
    return np.array(sequences), np.array(labels)

window_size = 60
X, y = create_sequences(scaled_data, window_size)

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(window_size, 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))

#Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')
print()
model.summary()

# Split the data
split = int(0.1 * len(X))
X_train, X_test = X[:], X[len(X)-split:]
y_train, y_test = y[:], y[len(X)-split:]

# Train the model
history = model.fit(X_train, y_train, epochs = 4, batch_size = 32, validation_data=(X_test, y_test))

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')

pred = model.predict(X_train)
rmse = np.sqrt(np.mean((pred - y_train)**2))
print('Root Mean Squared Error', rmse)
# Make predictions
predictions = model.predict(X_test)
rmse = np.sqrt(np.mean((predictions - y_test)**2))
print('Root Mean Squared Error', rmse)
predictions = scaler.inverse_transform(predictions)


# Debugging: Print the prediction
print(f"Prediction: {predictions[-1][0]}")

import pickle

with open('gold_price_prediction.pkl','wb') as file:
    pickle.dump(model,file)
    
with open('scaler.pkl','wb') as file:
    pickle.dump(scaler,file)
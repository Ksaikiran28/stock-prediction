import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input
from keras.optimizers import Adam
import joblib

# Load dataset
df = pd.read_csv('powergrid.csv')

# Display column names for debugging
print("Columns in CSV:", df.columns)

# Ensure 'Close' column exists
if 'Close' not in df.columns:
    raise ValueError("❌ 'Close' column not found. Please check your CSV file.")

# Clean and prepare the data
data = df[['Close']].dropna()  # Use only the 'Close' column and drop missing values
data['Close'] = pd.to_numeric(data['Close'], errors='coerce')  # Convert to numbers
data = data.dropna()  # Drop rows with NaN after conversion

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Save the scaler for later use
joblib.dump(scaler, 'scaler.save')

# Create the training dataset
training_data_len = int(np.ceil(len(scaled_data) * 0.8))

train_data = scaled_data[0:training_data_len, :]

x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

# Convert to numpy arrays and reshape
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Define the LSTM model
model = Sequential([
    Input(shape=(60, 1)),
    LSTM(50, return_sequences=True),
    LSTM(50, return_sequences=False),
    Dense(25),
    Dense(1)
])

# Compile the model
model.compile(optimizer=Adam(), loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)

# Save the model
model.save('stock_dl_model.h5')

print("✅ Model retrained and saved as 'stock_dl_model.h5'")

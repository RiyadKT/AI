# Importing necessary libraries
import pandas as pd
import tensorflow as tf
import numpy as np 
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# File path to the dataset
file_path = 'data.csv'

# Loading the dataset into a pandas DataFrame
data_frame = pd.read_csv(file_path)

# Extracting features (TIME) and target variable (Value) from the DataFrame
features = data_frame[['TIME']].values.astype(float)
target = data_frame['Value'].values.astype(float)

# Printing the features and target variable
print(data_frame[['TIME']].values.astype(float))
print(data_frame['Value'].values.astype(float))

# Printing the file path
print(file_path)

# Asking user for the name of the area
area_name = input("Enter the name of the area: ")
print("You entered:", area_name)

# Asking user for the units
units = input("Enter the units (per sq ft, per sq meter, per 100 sq meter): ")
print(f"You entered units: {units}")  # Keeping the print statement unchanged, but syntactically correct

# Scaling the features and target variable using MinMaxScaler
scaler_x = MinMaxScaler()
features_scaled = scaler_x.fit_transform(features)
scaler_y = MinMaxScaler()
target_scaled = scaler_y.fit_transform(target.reshape(-1, 1))

# Defining the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear')
])

# Compiling the model
model.compile(optimizer='adam', loss='mean_squared_logarithmic_error')

# Fitting the model to the scaled data
model.fit(features_scaled, target_scaled, epochs=1000, verbose=0)

# Predicting on the training data
training_predictions_scaled = model.predict(features_scaled)
training_predictions = scaler_y.inverse_transform(training_predictions_scaled).flatten()

# Plotting actual vs predicted values for training data
plt.scatter(data_frame['TIME'], data_frame['Value'].astype(float), label='Actual')
plt.scatter(data_frame['TIME'], training_predictions, label='Predicted (Training)', color='red')
plt.xlabel('Year')
plt.ylabel('₹')
plt.title(area_name)
plt.legend()
plt.show()

# Generating predictions for future years
future_years = np.arange(2023, 2030).reshape(-1, 1)
future_years_scaled = scaler_x.transform(future_years)
future_values_scaled = model.predict(future_years_scaled)
future_values = scaler_y.inverse_transform(future_values_scaled).flatten()

# Plotting actual vs predicted values for training and future data
plt.scatter(data_frame['TIME'], data_frame['Value'], label='Actual')
plt.scatter(data_frame['TIME'], training_predictions, label='Predicted (Training)', color='red')
plt.scatter(future_years.flatten(), future_values, label='Predicted (Future)', color='green')
plt.xlabel('Year')
plt.ylabel('₹')
plt.title(area_name)
plt.legend()
plt.show()

# Printing predicted values for future years
print('Predicted values:')
for year, value in zip(future_years.flatten(), future_values):
    formatted_value = "{:.2f}".format(value)
    print(f'Year {year}, Predicted Price: ₹{formatted_value}{units}')

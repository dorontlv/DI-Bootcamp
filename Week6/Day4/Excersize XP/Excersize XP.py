'''
Exercises XP


What You"ll learn
How to import and manipulate time-series data using pandas.
Techniques for handling missing values in time-series data.
Basic data visualization using matplotlib and seaborn.
Building and training a simple LSTM model for time-series data analysis.


What you will create
A cleaned and preprocessed time-series dataset.
Visualizations of the time-series data.
A simple LSTM model to analyze and predict time-series data.


Dataset
You will use this Dataset : household_power_consumption. You can find a description of the data here.


'''
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''


Exercise 1 : Data Import and Initial Exploration
Import the necessary libraries for data analysis and visualization.
Load the time-series dataset from the provided file.
Display the first few rows of the dataset to understand its structure.
Check the data types of each column and the shape of the dataset.

'''


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU
from keras.optimizers import SGD
import math
from sklearn.metrics import mean_squared_error

# Load the dataset text file
# The dataset is a TXT file containing time-series data related to household power consumption.
dataset = pd.read_csv('household_power_consumption.txt', sep=';', low_memory=False, na_values=['?'])  # treat '?' as NaN

# Display the first few rows of the dataset to understand its structure
print(dataset.head())

# Check the data types of each column and the shape of the dataset
print(dataset.dtypes)
print(dataset.shape)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''


Exercise 2 : Handling Missing Values
Identify columns in the dataset that contain missing values.
Fill the missing values using the mean of the respective columns.
Verify that there are no more missing values in the dataset.

'''

# Check for missing values in the dataset
print(dataset.isnull().sum())

# Identify columns with missing values
missing_columns = dataset.columns[dataset.isnull().any()]
print("Columns with missing values:", missing_columns)

# Fill missing values with the mean of respective columns
for column in missing_columns:
    dataset[column].fillna(dataset[column].mean(), inplace=True)

# Verify that there are no more missing values
print(dataset.isnull().sum())



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

'''

Exercise 3 : Data Visualization
Resample the "Global_active_power" column over a day and plot the sum and mean values.
Create a plot showing the mean and standard deviation of the "Global_intensity" column resampled over a day.

'''


# Convert the 'Date' and 'Time' columns to a single datetime column
dataset['Datetime'] = pd.to_datetime(dataset['Date'] + ' ' + dataset['Time'], format='%d/%m/%Y %H:%M:%S')

# Set the 'Datetime' column as the index
dataset.set_index('Datetime', inplace=True)

print(dataset.head())

# Resample the 'Global_active_power' column over a day and calculate sum and mean
daily_sum = dataset['Global_active_power'].resample('D').sum()
daily_mean = dataset['Global_active_power'].resample('D').mean()  # the mean value is a very small value

# Plot the sum and mean values
plt.figure(figsize=(12, 6))
plt.plot(daily_sum, label='Daily sum', color='blue')
plt.plot(daily_mean, label='Daily mean', color='orange')
plt.title('Daily sum and mean of global active power')
plt.xlabel('Date')
plt.ylabel('Global active power')
plt.legend()
plt.grid()
plt.show()


# Resample the 'Global_intensity' column over a day and calculate mean and standard deviation
daily_mean_intensity = dataset['Global_intensity'].resample('D').mean()
daily_std_intensity = dataset['Global_intensity'].resample('D').std()

# Plot the mean and standard deviation
plt.figure(figsize=(12, 6))
plt.plot(daily_mean_intensity, label='Daily mean', color='green')
plt.plot(daily_std_intensity, label='Daily standard deviation', color='red')
plt.title('Daily mean and standard deviation of global intensity')
plt.xlabel('Date')
plt.ylabel('Global intensity')
plt.legend()
plt.grid()
plt.show()



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

'''

Exercise 4 : Data Preprocessing for LSTM
Normalize the dataset to prepare it for LSTM model training.
Split the dataset into training and testing sets.
Reshape the data to fit the input requirements of an LSTM model.

'''

# let's say that we want to predict the next value of 'Global_active_power' , based on the previous 60 values.

# Normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset['Global_active_power'].values.reshape(-1, 1))  # convert into a 2D array

# Split the dataset into training and testing sets
training_size = int(len(scaled_data) * 0.80)
train_data = scaled_data[:training_size]
test_data = scaled_data[training_size:]

# Create sequences for LSTM input
def create_sequences(data, sequence_length):
    x, y = [], []
    for i in range(len(data) - sequence_length):
        x.append(data[i:i + sequence_length])  # for example - the last 60 days
        y.append(data[i + sequence_length])  # the current day
    return np.array(x), np.array(y)

sequence_length = 60  # Use the past 60 days
x_train, y_train = create_sequences(train_data, sequence_length)
x_test, y_test = create_sequences(test_data, sequence_length)

# Reshape the data to fit the LSTM input requirements
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''


Exercise 5 : Building an LSTM Model
Import the necessary libraries for building an LSTM model.
Define the architecture of the LSTM model, including the number of layers and neurons.
Compile the model with an appropriate loss function and optimizer.
'''

# Define the architecture of the LSTM model
model = Sequential()

# Add the first LSTM layer with Dropout regularization
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))

# Add a second LSTM layer with Dropout regularization
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

# Add a third LSTM layer with Dropout regularization
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))

# Add the output layer
model.add(Dense(units=1))  # Predicting the next value

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Display the model summary
model.summary()



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''


Exercise 6 : Training and Evaluating the LSTM Model
Train the LSTM model on the training dataset.
Evaluate the model's performance on the testing dataset.
Plot the training and validation loss to assess the model's learning progress.

'''


# Train the LSTM model on the training dataset
history = model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_test, y_test), verbose=1)

# Evaluate the model's performance on the testing dataset
test_loss = model.evaluate(x_test, y_test, verbose=1)
print("Test Loss:", test_loss)

# Plot the training and validation loss
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()


# Predict the future values using the trained model
predicted_data = model.predict(x_test)


# Plot the predicted data against the actual data
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual Data', color='blue')
plt.plot(predicted_data, label='Predicted Data', color='orange')
plt.title('Actual vs Predicted Data')
plt.xlabel('Time Steps')
plt.ylabel('Global Active Power')
plt.legend()
plt.grid()
plt.show()






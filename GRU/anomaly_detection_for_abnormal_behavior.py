import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount('/content/drive')

# Reading the data in CSV using Pandas
df = pd.read_csv('/content/drive/MyDrive/2_wheeler_related/DATASET-FINAL/Right_Deviation/ride_3.csv', parse_dates=['Time'])
df['Time'] = df['Time'].dt.tz_localize(None)
df.set_index('Time', inplace=True)
print(df.index[:5])
data = df[['Accel_X', 'Accel_Y','Accel_Z','Gyro_X', 'Gyro_Y','Gyro_Z','Yaw','Roll','Pitch']]

len(data)

# Window for time series analysis
def df_to_X_y(df, window_size=5):
    df_as_np = df.to_numpy()
    X = []
    y = []
    for i in range(len(df_as_np) - window_size):
        window_data = df_as_np[i:i + window_size]
        X.append(window_data)
        label = df_as_np[i + window_size]
        y.append(label)
    return np.array(X), np.array(y)

WINDOW_SIZE =  9
X1, y1 = df_to_X_y(data, WINDOW_SIZE)
X1.shape, y1.shape

print(f'Given the Array: \n{X1[0].flatten()}')
print(f'Predict this y: \n {y1[0]}')

# Split the data
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MinMaxScaler
total_samples = len(X1)
train_samples = int(0.7 * total_samples)
val_samples = int(0.15 * total_samples)
test_samples = total_samples - train_samples - val_samples

# Split the data into train, validation, and test sets
X_train, y_train = X1[:train_samples], y1[:train_samples]
X_val, y_val = X1[train_samples:train_samples + val_samples], y1[train_samples:train_samples + val_samples]
X_test, y_test = X1[train_samples + val_samples:], y1[train_samples + val_samples:]

# # Reshape X_train before normalization
# # Reshape and normalize X_train
# scaler_X = MinMaxScaler()
# samples, window_size, features = X_train.shape
# X_train_reshaped = X_train.reshape(samples, -1)  # Reshape to (number_of_samples, WINDOW_SIZE * 9)
# X_train_normalized = scaler_X.fit_transform(X_train_reshaped).reshape(samples, window_size, features)  # Reshape back to (number_of_samples, WINDOW_SIZE, 9)

# # # Similarly, reshape and normalize X_val and X_test
# X_val_reshaped = X_val.reshape(X_val.shape[0], -1)
# X_val_normalized = scaler_X.transform(X_val_reshaped).reshape(X_val.shape[0], window_size, features)

# X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
# X_test_normalized = scaler_X.transform(X_test_reshaped).reshape(X_test.shape[0], window_size, features)



# # # Normalize output labels
# scaler_y = MinMaxScaler()
# y_train_normalized = scaler_y.fit_transform(y_train)
# y_val_normalized = scaler_y.transform(y_val)
# y_test_normalized = scaler_y.transform(y_test)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import *
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import TimeSeriesSplit

# GRU Architecture
model1 = Sequential()
model1.add(InputLayer((WINDOW_SIZE, 9),))
model1.add(GRU(64, input_shape=(WINDOW_SIZE, 9), return_sequences=True))
model1.add(GRU(units=32, return_sequences=True))
model1.add(GRU(units=16))
model1.add(Dense(9, activation='relu'))
model1.add(Dense(9, activation='LeakyReLU'))
model1.compile(loss='mean_absolute_error', optimizer=Adam(learning_rate=0.001), run_eagerly=True)
model1.summary()

# Define the number of splits for time series cross-validation
n_splits = 4  # Can adjust this based on your preference
tscv = TimeSeriesSplit(n_splits=n_splits)
mae_scores = []

for train_index, test_index in tscv.split(X1):
    X_train, X_test = X1[train_index], X1[test_index]
    y_train, y_test = y1[train_index], y1[test_index]
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)

model1.fit(X_train, y_train, epochs=50)  # Set verbose to 1 for debugging

y_pred = model1.predict(X_test)

# MAE values 
mae = mean_absolute_error(y_test, y_pred)
mae_scores.append(mae)
mean_mae = np.mean(mae_scores)
print(f"Mean Cross-Validated MAE: {mean_mae}")

y_train.shape

train_predictions = model1.predict(X_train).flatten()
y_train = y_train.flatten()

train_results = pd.DataFrame(data={'Train Predictions': train_predictions, 'Actuals': y_train})
print(train_results)

train_predictions = model1.predict(X_train)
train_predictions.shape

num_outputs = train_predictions.shape[1]


# Defining output labels
output_labels = ['accelerometerAccelerationX(G)', 'accelerometerAccelerationY(G)', 'accelerometerAccelerationZ(G)',
                 'gyroRotationX(rad/s)', 'gyroRotationY(rad/s)', 'gyroRotationZ(rad/s)',
                 'motionYaw(rad)', 'motionRoll(rad)', 'motionPitch(rad)']

# Visualize each output separately with corresponding labels
for i in range(num_outputs):
    # Create a DataFrame with predicted and actual values for the current output
    output_data = pd.DataFrame(data={'Predicted': train_predictions.flatten()[:150], 'Actual': y_train.flatten()[:150]})
    output_data['Residual'] = output_data['Predicted'] - output_data['Actual']
    output_data['RMSE'] = np.sqrt(np.square(output_data['Residual']))


    mse = (output_data['Residual'] ** 2).mean()
    rmse = np.sqrt(mse)
    print(output_labels[i],rmse)
    # Plot predicted and actual values
    # plt.figure(figsize=(8, 4))
    # plt.plot(output_data['Predicted'], label='Predicted')
    # plt.plot(output_data['Actual'], label='Actual')
    # plt.xlabel('Sample Index')
    # plt.ylabel('Value')
    # plt.title(f'{output_labels[i]} Predicted vs Actual')
    # plt.legend()
    # plt.show()

val_predictions = model1.predict(X_val)
val_predictions.shape

num_outputs = val_predictions.shape[1]

# Define output labels
output_labels = ['accelerometerAccelerationX(G)', 'accelerometerAccelerationY(G)', 'accelerometerAccelerationZ(G)',
                 'gyroRotationX(rad/s)', 'gyroRotationY(rad/s)', 'gyroRotationZ(rad/s)',
                 'motionYaw(rad)', 'motionRoll(rad)', 'motionPitch(rad)']

# Visualize each output separately with corresponding labels
for i in range(num_outputs):
    # Create a DataFrame with predicted and actual values for the current output
    output_data = pd.DataFrame(data={'Predicted': val_predictions[:, i][:150], 'Actual': y_val[:, i][:150]})

# Plot predicted and actual values
    plt.figure(figsize=(8, 4))
    plt.plot(output_data['Predicted'], label='Predicted')
    plt.plot(output_data['Actual'], label='Actual')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.title(f'{output_labels[i]} Predicted vs Actual')
    plt.legend()
    plt.show()


test_predictions = model1.predict(X_test)
test_predictions.shape

num_outputs = test_predictions.shape[1]

# Define output labels
output_labels = ['accelerometerAccelerationX(G)', 'accelerometerAccelerationY(G)', 'accelerometerAccelerationZ(G)',
                 'gyroRotationX(rad/s)', 'gyroRotationY(rad/s)', 'gyroRotationZ(rad/s)',
                 'motionYaw(rad)', 'motionRoll(rad)', 'motionPitch(rad)']

# Visualize each output separately with corresponding labels
for i in range(num_outputs):
    # Create a DataFrame with predicted and actual testues for the current output
    output_data = pd.DataFrame(data={'Predicted': test_predictions[:, i][:150], 'Actual': y_test[:, i][:150]})

# Plot predicted and actual testues
    plt.figure(figsize=(8, 4))
    plt.plot(output_data['Predicted'], label='Predicted')
    plt.plot(output_data['Actual'], label='Actual')
    plt.xlabel('Sample Index')
    plt.ylabel('testue')
    plt.title(f'{output_labels[i]} Predicted vs Actual')
    plt.legend()
    plt.show()

# Calculate and print RMSE
    output_data['Residual'] = output_data['Predicted'] - output_data['Actual']
    output_data['Squared Residual'] = np.square(output_data['Residual'])
    output_data['RMSE'] = np.sqrt(output_data['Squared Residual'])
    rmse = output_data['RMSE'].mean()
    print(f"RMSE for {output_labels[i]}: {rmse}")

from sklearn.metrics import mean_squared_error, mean_absolute_error

# Assuming you already have X2 and y2 defined
train_predictions = model1.predict(X_test)
output_labels = ['accelerometerAccelerationX(G)', 'accelerometerAccelerationY(G)', 'accelerometerAccelerationZ(G)',
                 'gyroRotationX(rad/s)', 'gyroRotationY(rad/s)', 'gyroRotationZ(rad/s)',
                 'motionYaw(rad)', 'motionRoll(rad)', 'motionPitch(rad)']

all_residuals = []  # List to store residuals for each output

# Dictionary to store anomaly percentages for each output
anomaly_percentages = {}

for i in range(y_test.shape[1]):
    # Select the column corresponding to the current output from test_predictions
    predicted_column = test_predictions[:, i]

    # Select the column corresponding to the current output from y2
    actual_column = y_test[:, i]

    # Calculate residuals in vector form
    residuals = predicted_column - actual_column
    all_residuals.append(residuals)

    # Calculate the percentage of anomalies based on a threshold
    threshold = 0.5  # Adjust the threshold based on your requirements
    anomalies = (np.abs(residuals) > threshold).astype(int)
    anomaly_percentage = np.mean(anomalies) * 100
    anomaly_percentages[output_labels[i]] = anomaly_percentage

    # ... (rest of the plotting and visualization code)

# Convert the list of residuals into a NumPy array for further analysis if needed
all_residuals = np.array(all_residuals)

# Visualize the residuals in vector form
plt.figure(figsize=(10, 6))
plt.plot(all_residuals.T, label=output_labels)
plt.xlabel('Sample Index')
plt.ylabel('Residual Value')
plt.title('Residuals - Predicted vs Actual (Vector Form)')
plt.legend()
plt.show()

# Print anomaly percentages
for output_label, percentage in anomaly_percentages.items():
    print(f"Anomaly Percentage for {output_label}: {percentage:.2f}%")

# Evaluate the model
mae_test_per_output = [mean_absolute_error(y_test[:, i], test_predictions[:, i]) for i in range(test_predictions.shape[1])]
overall_mae_test = np.mean(mae_test_per_output)

mae_val_per_output = [mean_absolute_error(y_val[:, i], val_predictions[:, i]) for i in range(val_predictions.shape[1])]
overall_mae_val = np.mean(mae_val_per_output)


print(f'Overall Mean Absolute Error of test: {overall_mae_test}')
print(f'Overall Mean Absolute Error of val: {overall_mae_val}')

print("y_train shape:", y_train.shape)
print("train_predictions shape:", train_predictions.shape)

y_train_reshape = y_train.reshape(-1, 1)  # Reshape to have one column
mae_train_per_output = [np.abs(y_train_reshape - train_predictions[:, i]).mean() for i in range(train_predictions.shape[1])]
overall_mae_train = np.mean(mae_train_per_output)
print(f'Overall Mean Absolute Error of train: {overall_mae_train}')

# Assuming test_predictions is a 1D array and you want to reshape it
num_columns = y_train.shape[]
train_predictions = train_predictions.reshape(-1, num_columns)

# Evaluate the model
mae_train_per_output = [mean_absolute_error(y_test[:, i], train_predictions[:, i]) for i in range(num_columns)]
overall_mae_train = np.mean(mae_train_per_output)

print(f'Overall Mean Absolute Error of test: {overall_mae_train}')

# Assuming you have already defined train_predictions, y_train, val_predictions, and y_val
# train_predictions, y_train, val_predictions, y_val = ...

# Reshape y_train and y_val to match the number of output dimensions
y_train_reshaped = y_train.reshape(-1, num_outputs)
y_val_reshaped = y_val.reshape(-1, num_outputs)
y_test_reshaped = y_test.reshape(-1, num_outputs)

# Ensure that y_train_reshaped and train_predictions have the same number of samples
min_samples = min(y_train_reshaped.shape[0], train_predictions.shape[0])
y_train_reshaped = y_train_reshaped[:min_samples, :]
train_predictions = train_predictions[:min_samples, :]

# Ensure that y_val_reshaped and val_predictions have the same number of samples
min_samples_val = min(y_val_reshaped.shape[0], val_predictions.shape[0])
y_val_reshaped = y_val_reshaped[:min_samples_val, :]
val_predictions = val_predictions[:min_samples_val, :]

# Ensure that y_test_reshaped and test_predictions have the same number of samples
min_samples_test = min(y_test_reshaped.shape[0], test_predictions.shape[0])
y_test_reshaped = y_test_reshaped[:min_samples_test, :]
test_predictions = test_predictions[:min_samples_test, :]


# Calculate mean absolute error for each output dimension
mae_train_per_output = [mean_absolute_error(y_train_reshaped[:, i], train_predictions[:, i]) for i in range(train_predictions.shape[1])]
mae_val_per_output = [mean_absolute_error(y_val_reshaped[:, i], val_predictions[:, i]) for i in range(val_predictions.shape[1])]
mae_test_per_output = [mean_absolute_error(y_test_reshaped[:, i], test_predictions[:, i]) for i in range(test_predictions.shape[1])]

# Calculate overall mean absolute error
overall_mae_train = np.mean(mae_train_per_output)
overall_mae_val = np.mean(mae_val_per_output)
overall_mae_test = np.mean(mae_test_per_output)

print(f'Overall Mean Absolute Error of train: {overall_mae_train}')
print(f'Overall Mean Absolute Error of val: {overall_mae_val}')
print(f'Overall Mean Absolute Error of test: {overall_mae_test}')




# TESTING

df1 = pd.read_csv('/content/drive/MyDrive/2_wheeler_related/DATASET-FINAL/Right_Deviation/ride_9.csv', parse_dates=['Time'])
df1['lTime'] = df1['Time'].dt.tz_localize(None)
df1.set_index('lTime', inplace=True)
print(df.index[:5])

data = df[['Accel_X', 'Accel_Y','Accel_Z','Gyro_X', 'Gyro_Y','Gyro_Z','Yaw','Roll','Pitch']]

WINDOW_SIZE = 9
X2, y2 = df_to_X_y(data, WINDOW_SIZE)

from sklearn.metrics import mean_squared_error, mean_absolute_error

# Assuming you already have X2 and y2 defined
test_predictions = model1.predict(X2)
output_labels = ['accelerometerAccelerationX(G)', 'accelerometerAccelerationY(G)', 'accelerometerAccelerationZ(G)',
                 'gyroRotationX(rad/s)', 'gyroRotationY(rad/s)', 'gyroRotationZ(rad/s)',
                 'motionYaw(rad)', 'motionRoll(rad)', 'motionPitch(rad)']

for i in range(y2.shape[1]):
    # Select the column corresponding to the current output from test_predictions
    predicted_column = test_predictions[:, i]

    # Select the column corresponding to the current output from y2
    actual_column = y2[:, i]

    # Create a DataFrame with predicted and actual values for the current output
    output_data = pd.DataFrame(data={'Predicted': predicted_column.flatten(), 'Actual': actual_column.flatten()})

    # Plot predicted and actual values
    plt.figure(figsize=(8, 4))
    plt.plot(output_data['Predicted'][:150], label='Predicted')
    plt.plot(output_data['Actual'][:150], label='Actual')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.title(f'{output_labels[i]} - Predicted vs Actual')
    plt.legend()
    plt.show()

    # Calculate and print RMSE
    output_data['Residual'] = output_data['Predicted'] - output_data['Actual']
    output_data['Squared Residual'] = np.square(output_data['Residual'])
    output_data['RMSE'] = np.sqrt(output_data['Squared Residual'])
    rmse = output_data['RMSE'].mean()
    print(f"RMSE for {output_labels[i]}: {rmse}")

    # Residual plot
    plt.figure(figsize=(10, 6))
    plt.scatter(output_data.index, output_data['RMSE'], color='blue', label='Residuals')
    plt.axhline(y=0, color='red', linestyle='--', label='Zero Residual Line')
    plt.axhline(y=0.5, color='green', linestyle='--', label='Threshold Line at 0.5')  # Threshold line at 0.5
    plt.xlabel('Index')
    plt.ylabel('Residual')
    plt.title( 'Residual Plot')
    plt.legend()
    plt.show()

from sklearn.metrics import mean_squared_error, mean_absolute_error

# Assuming you already have X2 and y2 defined
test_predictions = model1.predict(X2)
output_labels = ['accelerometerAccelerationX(G)', 'accelerometerAccelerationY(G)', 'accelerometerAccelerationZ(G)',
                 'gyroRotationX(rad/s)', 'gyroRotationY(rad/s)', 'gyroRotationZ(rad/s)',
                 'motionYaw(rad)', 'motionRoll(rad)', 'motionPitch(rad)']

all_residuals = []  # List to store residuals for each output

for i in range(y2.shape[1]):
    # Select the column corresponding to the current output from test_predictions
    predicted_column = test_predictions[:, i]

    # Select the column corresponding to the current output from y2
    actual_column = y2[:, i]

    # Calculate residuals in vector form
    residuals = predicted_column - actual_column
    all_residuals.append(residuals)

    # Create a DataFrame with predicted and actual values for the current output
    output_data = pd.DataFrame(data={'Predicted': predicted_column.flatten(), 'Actual': actual_column.flatten(), 'Residual': residuals.flatten()})

    # Plot predicted and actual values
    # ... (rest of the plotting code)

    # Calculate and print RMSE
    # ... (rest of the RMSE calculation code)

    # Residual plot
    # ... (rest of the residual plot code)

# Convert the list of residuals into a NumPy array for further analysis if needed
all_residuals = np.array(all_residuals)

# Visualize the residuals in vector form
plt.figure(figsize=(10, 6))
plt.plot(all_residuals.T, label=output_labels)
plt.xlabel('Sample Index')
plt.ylabel('Residual Value')
plt.title('Residuals - Predicted vs Actual (Vector Form)')
plt.legend()
plt.show()

# Assuming you have test_predictions and y2 defined

# Calculate residuals between predicted and actual outputs
all_residuals = test_predictions - y2

# Define a threshold for anomaly detection
threshold = 0.5  # Change this value based on your analysis

# Identify anomalies based on the threshold
anomalies = np.abs(all_residuals) > threshold

# Calculate the percentage of anomalies detected
percentage_anomalies = np.mean(anomalies) * 100

print(f"Percentage of anomalies detected: {percentage_anomalies:.2f}%")

import numpy as np

all_residual_values = np.zeros((y2.shape[0],))

for i in range(y2.shape[0]):
    vector_magnitude_test_predictions = np.linalg.norm(test_predictions[i, :])
    vector_magnitude_y_test = np.linalg.norm(y2[i, :])

    all_residual_values[i] = vector_magnitude_test_predictions - vector_magnitude_y_test

print(all_residual_values)

import numpy as np
import matplotlib.pyplot as plt

# Set up the figure
plt.figure(figsize=(10, 6))

# Scatter plot of residuals against index
plt.scatter(np.arange(len(all_residual_values)), all_residual_values, color='blue', label='Residuals', alpha=0.5)

# Horizontal lines
plt.axhline(y=0, color='red', linestyle='--', label='Zero Residual Line')
plt.axhline(y=0.5, color='green', linestyle='--', label='Threshold Line at 0.5')

# Labels and title
plt.xlabel('Index')
plt.ylabel('Residual')
plt.title('Residual Plot Over Index')

# Legend
plt.legend()

# Show the plot
plt.show()

absolute_residual_values = np.abs(all_residual_values)

#Absolute 
import numpy as np
import matplotlib.pyplot as plt

# Set up the figure
plt.figure(figsize=(10, 6))

# Scatter plot of residuals against index
plt.scatter(np.arange(len(absolute_residual_values)), absolute_residual_values, color='blue', label='Residuals', alpha=0.5)

# Horizontal lines
plt.axhline(y=0, color='red', linestyle='--', label='Zero Residual Line')
plt.axhline(y=0.5, color='green', linestyle='--', label='Threshold Line at 0.65')

# Labels and title
plt.xlabel('Index')
plt.ylabel('Residual')
plt.title('Residual Plot Over Index')

# Legend
plt.legend()

# Show the plot
plt.show()

# Assuming 'threshold' is the chosen threshold for anomalies
threshold = 0.5
anomaly_indicator = (absolute_residual_values > threshold).astype(int)
print(anomaly_indicator)

df1['Time'].shape




import matplotlib.pyplot as plt

# Assuming df1 is your DataFrame containing the label column and anomaly_indicator

# Check the lengths of df1.index and anomaly_indicator
if len(df1.index) != len(anomaly_indicator):
    # Handle the mismatch in lengths, such as by trimming or padding one of the arrays
    min_length = min(len(df1.index), len(anomaly_indicator))
    df1.index = df1.index[:min_length]
    anomaly_indicator = anomaly_indicator[:min_length]

# Plotting the actual anomalies
plt.scatter(df1.index, df1['Time'], color='blue', label='Actual Anomalies')

# Plotting the predicted anomalies
plt.scatter(df1.index, anomaly_indicator, color='red', marker='o', label='Predicted Anomalies')  # Use 'o' as marker instead of 'blue'

# Setting labels and title
plt.xlabel('Index')
plt.ylabel('Anomaly Indicator')
plt.title('Actual vs Predicted Anomalies')

# Displaying the legend
plt.legend()

# Show the plot
plt.show()

# Assuming df is your DataFrame
rows_to_remove = 8
random_rows_indices = np.random.choice(df1.index, rows_to_remove, replace=False)

# Remove the randomly selected rows in-place
df1.drop(random_rows_indices, inplace=True)

from sklearn.metrics import classification_report

# Assuming 'true_labels' is the true label column from your dataset
classification_report(df1['label(N)'], anomaly_indicator)

from sklearn.metrics import classification_report
import pandas as pd

# Assuming 'threshold' is the chosen threshold for anomalies
threshold = 0.4
anomaly_indicator = (absolute_residual_values > threshold).astype(int)
print(anomaly_indicator.shape)

# Assuming 'true_labels' is the true label column from your dataset
true_labels = df1['label(N)']

# Align the lengths of true_labels and anomaly_indicator
min_length = min(len(true_labels), len(anomaly_indicator))
true_labels = true_labels[:min_length]
anomaly_indicator = anomaly_indicator[:min_length]

# Generate classification report
report = classification_report(true_labels, anomaly_indicator)

# Split the classification report into lines
lines = report.split('\n')

# Extract class names and metrics
class_names = lines[0].split()
metrics = [line.split() for line in lines[2:-5]]

# Create a DataFrame
report_df = pd.DataFrame(metrics, columns=['Class'] + class_names)

# Print the formatted classification report
print(report_df)

total_anomalies = df1[df1['label(N)'].isin([1, 2])].shape[0]
print("Total number of anomalies:", total_anomalies)
num_detected_anomalies = anomaly_indicator.sum()
print("Number of anomalies detected:", num_detected_anomalies)

plt.figure(figsize=(10, 6))
plt.scatter(train_results.index, train_results['Residual'], color='blue', label='Residuals')
plt.axhline(y=0, color='red', linestyle='--', label='Zero Residual Line')
plt.xlabel('Index')
plt.ylabel('Residual')
plt.title('Residual Plot')
plt.legend()
plt.show()

test_data_pred['Residual'] = test_data_pred['Predicted'] - test_data_pred['Actual']

print(train_results['Actuals'][3779:3783])
print(train_results['Train Predictions'][3779:3783])
train_results['Residual'] = train_results['Train Predictions'] - train_results['Actuals']
print(train_results['Residual'][3779:3783])

filtered_results = train_results[(train_results['Residual'] > 0.5) | (train_results['Residual'] < -0.5)]
print(filtered_results)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
for WINDOW_SIZE in range(1, 9):
    X1, y1 = df_to_X_y(data, WINDOW_SIZE)

    total_samples = len(X1)
    train_samples = int(0.7 * total_samples)
    val_samples = int(0.15 * total_samples)
    test_samples = total_samples - train_samples - val_samples

    # Split the data
    X_train, y_train = X1[:train_samples], y1[:train_samples]
    X_val, y_val = X1[train_samples:train_samples + val_samples], y1[train_samples:train_samples + val_samples]
    X_test, y_test = X1[train_samples + val_samples:], y1[train_samples + val_samples:]

    model1 = Sequential()
    model1.add(InputLayer((WINDOW_SIZE, 9)))
    model1.add(Bidirectional(GRU(128, return_sequences=True, activation='relu', dropout=0.2, recurrent_dropout=0.2)))
    model1.add(Bidirectional(GRU(64, return_sequences=True, activation='relu', dropout=0.2, recurrent_dropout=0.2)))
    model1.add(Bidirectional(GRU(32, return_sequences=True, activation='relu', dropout=0.2, recurrent_dropout=0.2)))
    model1.add(Bidirectional(GRU(16, activation='relu', dropout=0.2, recurrent_dropout=0.2)))


# Dense layers
    model1.add(Dense(8, activation='relu'))
    model1.add(Dropout(0.2))
    model1.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=1e-3), metrics=[RootMeanSquaredError()])
# Output layer
    model1.add(Dense (9, activation='LeakyReLU'))
    model1.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10)

    train_predictions = model1.predict(X_train).flatten()
# Assuming train_predictions and y_train are 1D arrays
    train_results = pd.DataFrame(data={'Train Predictions': train_predictions.flatten(), 'Actuals': y_train.flatten()})

    val_predictions = model1.predict(X_val).flatten()
    val_results = pd.DataFrame(data={'Val Predictions': val_predictions.flatten(), 'Actuals': y_val.flatten()})

    test_predictions = model1.predict(X_test).flatten()
    test_results = pd.DataFrame(data={'Test Predictions': test_predictions.flatten(), 'Actuals': y_test.flatten()})

    plt.figure(figsize=(6, 6))
    plt.subplot(3, 1, 1)
    plt.plot(train_results['Actuals'][:150], label='Train Actuals')
    plt.plot(train_results['Train Predictions'][:150], label='Train Predictions')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(val_results['Actuals'][:150], label='Val Actuals')
    plt.plot(val_results['Val Predictions'][:150], label='Val Predictions')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(test_results['Actuals'][:150], label='Test Actuals')
    plt.plot(test_results['Test Predictions'][:150], label='Test Predictions')
    plt.legend()

    plt.suptitle(f'Window Size: {WINDOW_SIZE}')
    plt.show()

test_predictions = model.predict(X_test)
val_predictions = model.predict(X_val)
train_predictions = model.predict(X_train)

# For each output
output_colors = ['blue', 'yellow', 'brown', 'orange', 'red', 'pink', 'green', 'black', 'purple']

# Plotting
plt.figure(figsize=(12, 8))

# Iterate over each output
for i in range(9):
    plt.plot(y_test[:, i], label=f'Actual  {i+1}', color=output_colors[i], linestyle='-')
    plt.plot(test_predictions[:, i], label=f'Predicted  {i+1}', color=output_colors[i], linestyle='--')

plt.legend()
plt.title('Actual and Predicted Values for Each')
plt.xlabel('Sample')
plt.ylabel('Value')
plt.show()

# For test
# Calculate Mean Absolute Error (MAE)
mae_per_output = mean_absolute_error(y_test, test_predictions, multioutput='raw_values')
overall_mae = np.mean(mae_per_output)
for i, mae in enumerate(mae_per_output):
    print(f'MAE for Output {i+1}: {mae}')
print(f'Overall MAE: {overall_mae}')

# Calculate Mean Squared Error (MSE)
mse_per_output = mean_squared_error(y_test, test_predictions, multioutput='raw_values')
overall_mse = np.mean(mse_per_output)
for i, mse in enumerate(mse_per_output):
    print(f'MSE for Output {i+1}: {mae}')
print(f'Overall MSE: {overall_mse}')

# Calculate Root Mean Squared Error (RMSE) for each output
rmse_per_output = np.sqrt(mse_per_output)
overall_rmse = np.mean(rmse_per_output)
for i, rmse in enumerate(rmse_per_output):
    print(f'RMSE for Output {i+1}: {rmse}')
print(f'Overall RMSE: {rmse}') 
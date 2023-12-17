
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Conv1D, Flatten, LSTM, concatenate


def build_dnn_model(input_shape):
    """
    Builds a Dense Neural Network model.

    Args:
    input_shape (tuple): Shape of the input data (excluding batch size).

    Returns:
    keras.Model: A compiled Keras model.
    """
    inputs = Input(shape=input_shape)
    x = Dense(128, activation='relu')(inputs)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(1)(x)  # Assuming salary prediction is a regression problem
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def build_cnn_model(input_shape):
    """
    Builds a Convolutional Neural Network model.

    Args:
    input_shape (tuple): Shape of the input data (excluding batch size).

    Returns:
    keras.Model: A compiled Keras model.
    """
    inputs = Input(shape=input_shape)
    x = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def build_rnn_model(input_shape):
    """
    Builds a Recurrent Neural Network model.

    Args:
    input_shape (tuple): Shape of the input data (excluding batch size).

    Returns:
    keras.Model: A compiled Keras model.
    """
    inputs = Input(shape=input_shape)
    x = LSTM(64, return_sequences=True)(inputs)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def build_stacked_model(input_shape):
    """
    Builds a stacked model that combines DNN, CNN, and RNN models.

    Args:
    input_shape (tuple): Shape of the input data (excluding batch size).

    Returns:
    keras.Model: A compiled Keras model.
    """
    # Create individual models
    dnn = build_dnn_model(input_shape)
    cnn = build_cnn_model(input_shape)
    rnn = build_rnn_model(input_shape)

    # Combined stacked model
    combinedInput = concatenate([dnn.output, cnn.output, rnn.output])
    x = Dense(64, activation="relu")(combinedInput)
    x = Dense(1)(x)

    model = Model(inputs=[dnn.input, cnn.input, rnn.input], outputs=x)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

#setup data

data = pd.read_csv('/home/brandon/ML Zoomcamp/Capstone Project 2/salaries.csv')

data.head()

#Data EDA
import matplotlib.pyplot as plt
import seaborn as sns


# Display dataset information (column data types, non-null values, etc.)
print(data.info())

# Summary statistics for numeric columns
print(data.describe())


data.columns

# Check for missing values
print(data.isnull().sum())


# Histograms for numerical features
data.hist(bins=15, figsize=(15, 10))
plt.show()


# Count plots for categorical features for top 10 most frequent categories
categorical_cols = data.select_dtypes(include=['object']).columns

for col in categorical_cols:
    plt.figure(figsize=(15, 10))
    sns.countplot(y=col, data=data, order=data[col].value_counts().iloc[:10].index)
    plt.show()
    

# Heatmap of the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap='viridis')
plt.title("Correlation Matrix")
plt.show()


# Assuming 'salary' and 'remote_ratio' are numeric in your dataset
numeric_features = ['salary', 'remote_ratio']                
sns.pairplot(data=data, vars=numeric_features, hue='experience_level', palette='viridis')


# Scatter plot between 'salary' and 'remote_ratio'
plt.figure(figsize=(8, 6))
sns.scatterplot(x='salary', y='remote_ratio', data=data)
plt.title("Scatter Plot between Salary and Remote Ratio")
plt.show()



# Scatter plot between 'salary' and 'work_year'
plt.figure(figsize=(8, 6))
sns.scatterplot(x='salary', y='work_year', data=data)
plt.title("Scatter Plot between Salary and Work Year")
plt.show()



# Boxplot for a feature
plt.figure(figsize=(8, 6))
sns.boxplot(y='salary', data=data)
plt.title("Boxplot of Feature1")
plt.show()

# Violin plot for a feature
plt.figure(figsize=(8, 6))
sns.violinplot(y='salary', data=data)
plt.title("Violin Plot of Feature1")
plt.show()

# <h1>Model Testing<h1>


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

X = data.drop('salary_in_usd', axis=1)
y = data['salary_in_usd']

# Identify and encode categorical columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns
X_encoded = pd.get_dummies(X, columns=categorical_cols)

# Scale your features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


input_shape = X_train.shape[1:]  # Shape of input data excluding batch size
model = build_dnn_model(input_shape)


# Fit the model on the training data
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model on the test data
test_loss = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")


# Make predictions (optional)
predictions = model.predict(X_test)
print(predictions)


#Test the CNN model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Conv1D, Flatten
import pandas as pd
import numpy as np


# Split data into features and target
X = data.drop('salary_in_usd', axis=1)
y = data['salary_in_usd']

# Encode categorical columns and scale features
categorical_cols = X.select_dtypes(include=['object', 'category']).columns
X_encoded = pd.get_dummies(X, columns=categorical_cols)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Reshape X_train and X_test to add an additional dimension
X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Define the CNN model building function 
def build_cnn_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Build the CNN model
input_shape = X_train_reshaped.shape[1:]  # Shape of input data excluding batch size
model = build_cnn_model(input_shape)

# Fit the model on the training data
history = model.fit(X_train_reshaped, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model on the test data
test_loss = model.evaluate(X_test_reshaped, y_test)
print(f"Test Loss: {test_loss}")

# Make predictions (optional)
predictions = model.predict(X_test_reshaped)
print(predictions)

#Test the RNN model

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LSTM, Flatten
import pandas as pd
import numpy as np

# Split data into features and target
X = data.drop('salary_in_usd', axis=1)
y = data['salary_in_usd']

# Encode categorical columns and scale features
categorical_cols = X.select_dtypes(include=['object', 'category']).columns
X_encoded = pd.get_dummies(X, columns=categorical_cols)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Reshape X_train and X_test to add an additional dimension
X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Define the RNN model building function (assuming this is already defined)
def build_rnn_model(input_shape):
    inputs = Input(shape=input_shape)
    x = LSTM(64, return_sequences=True)(inputs)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Build the RNN model
input_shape = X_train_reshaped.shape[1:]  # Shape of input data excluding batch size
model = build_rnn_model(input_shape)

# Fit the model on the training data
history = model.fit(X_train_reshaped, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model on the test data
test_loss = model.evaluate(X_test_reshaped, y_test)
print(f"Test Loss: {test_loss}")

# Make predictions (optional)
predictions = model.predict(X_test_reshaped)
print(predictions)

#Test the Stacked model

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Conv1D, Flatten, LSTM, concatenate


X = data.drop('salary_in_usd', axis=1)
y = data['salary_in_usd']

# Encode categorical columns and scale features
categorical_cols = X.select_dtypes(include=['object', 'category']).columns
X_encoded = pd.get_dummies(X, columns=categorical_cols)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Reshape X_train and X_test to add an additional dimension
X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

#Define other models with flatten
def build_dnn_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Flatten()(inputs)  # Flatten the input
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    return Model(inputs, x)


def build_cnn_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)  # Conv1D layer
    x = Flatten()(x)  # Flatten the output
    x = Dense(64, activation='relu')(x)  # Dense layer
    return Model(inputs, x)

def build_rnn_model(input_shape):
    inputs = Input(shape=input_shape)
    x = LSTM(64, return_sequences=False)(inputs)  # LSTM layer
    x = Dense(64, activation='relu')(x)  # Dense layer
    return Model(inputs, x)



# Define the stacked model building function
def build_stacked_model(input_shape):
    # Create individual models with the same final layer units
    dnn = build_dnn_model(input_shape)
    cnn = build_cnn_model(input_shape)
    rnn = build_rnn_model(input_shape)

    # Combined stacked model
    combinedInput = concatenate([dnn.output, cnn.output, rnn.output])
    x = Dense(64, activation="relu")(combinedInput)
    x = Dense(1)(x)

    model = Model(inputs=[dnn.input, cnn.input, rnn.input], outputs=x)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Build the stacked model
input_shape = X_train_reshaped.shape[1:]  # Shape of input data excluding batch size
model = build_stacked_model(input_shape)

# Fit the model on the training data
history = model.fit([X_train_reshaped, X_train_reshaped, X_train_reshaped], y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model on the test data
test_loss = model.evaluate([X_test_reshaped, X_test_reshaped, X_test_reshaped], y_test)
print(f"Test Loss: {test_loss}")

# Make predictions (optional)
predictions = model.predict([X_test_reshaped, X_test_reshaped, X_test_reshaped])
print(predictions)

#What is the best model?

print(history.history.keys())
print(history.history['loss'])
print(history.history['val_loss'])
print(f'The best model is the stacked model with a loss of {min(history.history["val_loss"])}')

#Save the stacked model using pickle

import pickle
import os

# Save the model to pickle file
filename = 'stacked_model.pkl'
pickle.dump(model, open(filename, 'wb'))

# Load the model from the pickle file

with open(filename, 'rb') as file:
    pickle_model = pickle.load(file)
    
# Save the TensorFlow model
model.save('stacked_model.h5')






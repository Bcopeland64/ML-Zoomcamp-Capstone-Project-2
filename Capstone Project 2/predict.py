import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import load_model

# Function to preprocess data
def preprocess_data(input_data, categorical_cols):
    # Assume all non-categorical columns are numerical
    numerical_cols = [col for col in input_data.columns if col not in categorical_cols]

    # Create transformers
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())])

    # Combine into a preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)])

    # Apply transformations
    input_data_processed = preprocessor.fit_transform(input_data)
    return input_data_processed

# Function to load the trained model
def load_trained_model(model_path):
    return load_model(model_path)

# Function to make prediction
def make_prediction(model, processed_data):
    # Assume the first input is for DNN, second for CNN, third for RNN
    # Adjust the reshaping based on your model's input requirements
    input_for_dnn = processed_data.toarray()
    input_for_cnn = processed_data.toarray().reshape(-1, 316, 1)
    input_for_rnn = processed_data.toarray().reshape(-1, 316, 1)

    # Make prediction with three inputs
    return model.predict([input_for_dnn, input_for_cnn, input_for_rnn])

def main():
    # Load the model
    model = load_trained_model('/home/brandon/ML Zoomcamp/Capstone Project 2/stacked_model.h5')

    # Load some data to make predictions
    input_data = pd.read_csv('/home/brandon/ML Zoomcamp/Capstone Project 2/salaries.csv')
    
    # Drop the target variable if present
    if 'salary_in_usd' in input_data.columns:
        input_data = input_data.drop('salary_in_usd', axis=1)

    # Define categorical columns (update this as per your dataset)
    categorical_cols = [col for col in input_data.columns if input_data[col].dtype == 'object']

    # Preprocess data
    processed_data = preprocess_data(input_data, categorical_cols)

    # Make predictions
    predictions = make_prediction(model, processed_data)
    print(predictions)

if __name__ == "__main__":
    main()

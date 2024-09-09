import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from category_encoders import TargetEncoder


def load_data(filepath):
    # read in training data
    data = pd.read_csv(filepath)
    return data



def handle_missing_values(data):
    # drop columns 'clean_title' and 'id'
    data.drop('clean_title', axis=1, inplace=True)
    data.drop('id', axis=1, inplace=True)
    
    # Drop NaN rows where 'fuel_type' or 'accident' have missing values
    data.dropna(subset=['fuel_type', 'accident'], inplace=True)



def encode_categorical_values(data):

    # List of features to apply frequency or target encoding
    features_to_encode = ['brand', 'model', 'model_year', 'milage', 'fuel_type', 'engine', 'transmission', 'ext_col', 'int_col']

    # Replace values in the 'accident' column
    data['accident'] = data['accident'].map({
        'None reported': 0,
        'At least 1 accident or damage reported': 1
    })
    

    ### TARGET ENCODING ###

    # Initialize the target encoder
    target_encoder = TargetEncoder(cols=features_to_encode, smoothing=0.3)
    # Apply target encoding
    target_encoded = target_encoder.fit_transform(data[features_to_encode], data['price'])

    # Drop the original categorical columns
    data.drop(features_to_encode, axis=1, inplace=True)


    return target_encoded, data



def scale_numerical_features(data):
    numerical_features = ['brand', 'model', 'model_year', 'milage', 'fuel_type', 'engine', 'transmission', 'ext_col', 'int_col']
    scaler = StandardScaler()
    data[numerical_features] = scaler.fit_transform(data[numerical_features])



def combine_dataframes(data1, data2):
    # Reset indices of both DataFrames
    data1.reset_index(drop=True, inplace=True)
    data2.reset_index(drop=True, inplace=True)

    # Concatenate the Encoded Data with the Original DataFrame
    data_processed = pd.concat([data1, data2], axis=1)
    return data_processed


if __name__ == "__main__":
    data = load_data('data/raw/train.csv')
    handle_missing_values(data)
    target_encoded, data = encode_categorical_values(data)
    scale_numerical_features(target_encoded)

    baseline_data_processed = combine_dataframes(target_encoded, data)
    baseline_data_processed['price'].to_csv('data/processed/baseline_train_labels_processed.csv', index=False)
    baseline_data_processed.drop('price', axis=1, inplace=True)
    baseline_data_processed.to_csv('data/processed/baseline_train_data_processed.csv', index=False)
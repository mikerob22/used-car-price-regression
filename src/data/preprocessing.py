import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.features.feature_engineering import feature_engineering

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

    # List of features to apply frequency encoding
    features_to_encode = ['brand', 'model', 'engine', 'transmission', 'ext_col', 'int_col']

    ### FREQUENCY ENCODING ###
    for feature in features_to_encode:
        # Calculate frequency of each category
        freq_encoding = data[feature].value_counts() / len(data)
    
        # Map frequencies to the original feature
        data[feature] = data[feature].map(freq_encoding)


    # Replace values in the 'accident' column
    data['accident'] = data['accident'].map({
        'None reported': 0,
        'At least 1 accident or damage reported': 1
    })


    ### ONE-HOT ENCODING ###

    fuel_type = data[['fuel_type']]

    cat_encoder = OneHotEncoder(sparse_output=False)
    fuel_type_1hot = cat_encoder.fit_transform(fuel_type)

    # Convert the OneHotEncoded Data to a DataFrame
    fuel_type_1hot_df = pd.DataFrame(fuel_type_1hot, columns=cat_encoder.get_feature_names_out(['fuel_type']))

    data_encoded = data

    return data_encoded, fuel_type_1hot_df



def scale_numerical_features(data):
    numerical_features = ['model_year', 'milage', 'car_age']
    scaler = StandardScaler()
    data[numerical_features] = scaler.fit_transform(data[numerical_features])


def combine_dataframes(data1, data2):
    # Reset indices of both DataFrames
    data1.reset_index(drop=True, inplace=True)
    data2.reset_index(drop=True, inplace=True)

    # Concatenate the Encoded Data with the Original DataFrame
    data_processed = pd.concat([data1, data2], axis=1)

    # Step 4: Drop the Original 'fuel_type' Column
    data_processed.drop('fuel_type', axis=1, inplace=True)

    return data_processed

if __name__ == "__main__":
    data = load_data('data/raw/train.csv')
    handle_missing_values(data)
    data_encoded, data_one_hot = encode_categorical_values(data)
    feature_engineering(data_encoded)
    scale_numerical_features(data_encoded)
    data_processed = combine_dataframes(data_encoded, data_one_hot)
    
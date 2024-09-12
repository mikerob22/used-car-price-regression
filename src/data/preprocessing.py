import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from src.features.feature_engineering import feature_engineering
from category_encoders import TargetEncoder
import joblib


def load_data(filepath):
    # read in training data
    return pd.read_csv(filepath)


def handle_missing_values(data, train=True):
    if not train:
        # extract and save the 'id' column for later use
        id_column = data['id'].copy()
        # drop columns 'clean_title' and 'id'
        data.drop(['clean_title', 'id'], axis=1, inplace=True)
    
    else:
        data.drop(['clean_title', 'id'], axis=1, inplace=True) 
    # Drop NaN rows where 'fuel_type' or 'accident' have missing values
    data.dropna(subset=['fuel_type', 'accident'], inplace=True)

    if not train:
        return data, id_column

    return data


def impute_feature_engineered(data, imputer=None, train=True):
    columns_to_impute = ['horsepower', 'engine_size', 'power_to_weight_ratio']

    if train:
        # Initialize Iterative Imputer
        imputer = IterativeImputer(max_iter=10, random_state=42)
        # Apply imputation only on specified columns
        data[columns_to_impute] = imputer.fit_transform(data[columns_to_impute])
        joblib.dump(imputer, 'src/data/serialized/imputer.pkl')
    else:
        data[columns_to_impute] = imputer.transform(data[columns_to_impute])

    return data


def onehotencode_columns(data, columns, encoder=None, train=True):
    """
    One-hot encodes the specified columns of the DataFrame.

    Parameters:
    - data: pandas DataFrame
    - columns: list of column names to one-hot encode

    Returns:
    - pandas DataFrame with one-hot encoded columns
    """
    if train:
        # Initialize the OneHotEncoder
        encoder = OneHotEncoder(sparse_output=False, drop='first')  # drop='first' to avoid multicollinearity
        # Fit-transform the columns
        encoded_cols = encoder.fit_transform(data[columns])
        joblib.dump(encoder, 'src/data/serialized/onehot_encoder.pkl')
    else:
        encoded_cols = encoder.transform(data[columns])
    
    # Create DataFrame
    encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(columns))
    # Concatenate the original DataFrame with the encoded DataFrame
    data = data.drop(columns, axis=1)  # Drop the original columns
    data = pd.concat([data.reset_index(drop=True), encoded_df], axis=1)

    return data, encoder


def encode_categorical_values(data, target_column='price', encoder=None, train=True):

    # List of features to apply frequency or target encoding
    features_to_encode = ['brand', 'model', 'engine', 'ext_col', 'int_col']

    '''
    ### FREQUENCY ENCODING ###
    for feature in features_to_encode:
        # Calculate frequency of each category
        freq_encoding = data[feature].value_counts() / len(data)
    
        # Map frequencies to the original feature
        data[feature] = data[feature].map(freq_encoding)
    '''

    # Replace values in the 'accident' column
    data['accident'] = data['accident'].map({
        'None reported': 0,
        'At least 1 accident or damage reported': 1
    })
    
    if train:
        ### TARGET ENCODING ###
        # Initialize the target encoder
        encoder = TargetEncoder(cols=features_to_encode, smoothing=0.3)
        # Apply target encoding
        target_encoded = encoder.fit_transform(data[features_to_encode], data[target_column])
        joblib.dump(encoder, 'src/data/serialized/target_encoder.pkl')
    else:
        target_encoded = encoder.transform(data[features_to_encode])

    # Drop the original categorical columns
    data.drop(features_to_encode, axis=1, inplace=True)

    return data, target_encoded, encoder


def preprocess_data(data, train=True):
    if train:
        data = handle_missing_values(data, train=train)
    else:
        data, id_column = handle_missing_values(data, train=train)
        
    data = feature_engineering(data)
    # print("Data after feature engineering:", data)

    if train:
        data = impute_feature_engineered(data, train=train)
        data, target_encoded, target_encoder = encode_categorical_values(data, train=train)
        data, onehot_encoder = onehotencode_columns(data, ['fuel_type', 'transmission'], train=train)
    else: 
        imputer = joblib.load('src/data/serialized/imputer.pkl')
        data = impute_feature_engineered(data, imputer=imputer, train=train)
        target_encoder = joblib.load('src/data/serialized/target_encoder.pkl')
        data, target_encoded, _ = encode_categorical_values(data, encoder=target_encoder, train=train)
        onehot_encoder = joblib.load('src/data/serialized/onehot_encoder.pkl')
        data, _ = onehotencode_columns(data, ['fuel_type', 'transmission'], encoder=onehot_encoder, train=train)
    
    if not train:
        # Reset index for both id_column and processed data
        id_column.reset_index(drop=True, inplace=True)
        data.reset_index(drop=True, inplace=True)
        # Combine the stored 'id' column with the processed data
        data = pd.concat([id_column, data], axis=1)

    return data



def scale_numerical_features(data):
    numerical_features = ['model_year', 'milage', 'car_age']
    scaler = StandardScaler()
    data[numerical_features] = scaler.fit_transform(data[numerical_features])


def scale_target_encoded_features(data):
    encoded_features = ['brand', 'model', 'engine', 'transmission', 'ext_col', 'int_col']
    scaler = StandardScaler()
    data[encoded_features] = scaler.fit_transform(data[encoded_features])



def combine_dataframes(data1, data2, data3):
    # Reset indices of both DataFrames
    data1.reset_index(drop=True, inplace=True)
    data2.reset_index(drop=True, inplace=True)
    data3.reset_index(drop=True, inplace=True)

    # Concatenate the Encoded Data with the Original DataFrame
    data_processed = pd.concat([data1, data2, data3], axis=1)

    # Step 4: Drop the Original 'fuel_type' Column
    data_processed.drop('fuel_type', axis=1, inplace=True)

    return data_processed



if __name__ == "__main__":
    # data = load_data('data/raw/train.csv')

    # handle_missing_values(data, train=True)

    # train_data_processed = preprocess_data(data, train=True)
    from src.visualization.visualize import plot_missing_values
    data = load_data('data/raw/test.csv')
    data, id_column = handle_missing_values(data, train=False)
    
    missing_value_counts = data.isna().sum()
    plot_missing_values(missing_value_counts)

    train_data_processed = preprocess_data(data, train=False)





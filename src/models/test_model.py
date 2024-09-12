import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from joblib import load
from src.data.preprocessing import load_data, preprocess_data

### RANDOM FOREST REGRESSOR MODEL ###

def test_rf_regressor(test_data):

    # Initialize and train the Random Forest Regressor
    rf_model = load('src/models/serialized/rf_model.pkl')


    # Make predictions
    y_pred = rf_model.predict(test_data)

    
    # Evaluate the model
    # mse = mean_squared_error(train_labels, y_pred)
    # rmse = np.sqrt(mse)
    # mae = mean_absolute_error(train_labels, y_pred)
    # r2 = r2_score(train_labels, y_pred)

    # print(f"Mean Squared Error: {mse}")
    # print(f"Root Mean Squared Error: {rmse}")
    # print(f"Mean Absolute Error: {mae}")
    # print(f"r2: {r2}")

    


if __name__ == '__main__':
    test_data = load_data('data/raw/test.csv')

    processed_test_data = preprocess_data(test_data, train=False)

    train_df = processed_train_data.drop('price', axis=1)
    # train_df.to_csv('data/processed/train_data_processed.csv')
    train_labels = processed_train_data['price']
    # train_labels.to_csv('data/processed/train_labels_processed.csv')
    
    
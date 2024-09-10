import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from joblib import load

### RANDOM FOREST REGRESSOR MODEL ###

def test_rf_regressor():

    # Initialize and train the Random Forest Regressor
    rf_model = load('src/models/serialized/rf_model.pkl')


    # Make predictions
    y_pred = rf_model.predict(train_data)

    # Evaluate the model
    mse = mean_squared_error(train_labels, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(train_labels, y_pred)
    r2 = r2_score(train_labels, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {rmse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"r2: {r2}")

    


if __name__ == '__main__':
    test_data = pd.read_csv('data/processed/test_data_processed.csv')
    test_labels = pd.read_csv('data/test_data/test_labels_processed.csv')
    rf_regressor_test_results = test_rf_regressor(test_data, test_labels)
    
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

    # Keep a copy of the 'id' column
    ids = test_data['id']

    test_data = test_data.drop(columns = ['id'])

    # Make predictions
    y_pred = rf_model.predict(test_data)

    submission_df = pd.DataFrame({
    'id': ids,
    'price': y_pred
    })

    # Step 4: Save the results to submission.csv
    submission_df.to_csv('data/submission.csv', index=False)
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
    
    test_rf_regressor(processed_test_data)
    
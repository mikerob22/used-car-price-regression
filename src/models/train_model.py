import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

train_data = pd.read_csv('data/processed/train_data_processed.csv', index_col=False)
train_labels = pd.read_csv('data/processed/train_labels_processed.csv', index_col=False)

# train_data.drop('index', axis=1, inplace=True)
train_labels = train_labels['price'] 


### BASELINE LINEAR REGRESSION MODEL ###
def linear_reg_model():
     # Add a constant to the model (the intercept term)
    train_data_with_const = sm.add_constant(train_data)

    # Initialize and train the model
    model = sm.OLS(train_labels, train_data_with_const).fit()

    # Make predictions
    y_pred = model.predict(train_data_with_const)

    # Evaluate the model
    mse = mean_squared_error(train_labels, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(train_labels, y_pred)
    r2 = r2_score(train_labels, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {rmse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"r2: {r2}")

    # Print the summary of the OLS regression results
    summary = model.summary()
    print(summary)

    # Save the summary to a text file
    with open('OLS_Regression_Summary.txt', 'w') as file:
        file.write(summary.as_text())


### ELASTIC NET REGRESSION MODEL W/ HYPERPARAMETER TUNING ###
def elastic_net_model():
    # Initialize and train the model
    model = ElasticNet()

    # Setting up the parameter grid
    param_grid = {
        'alpha': [0.1, 1.0, 10.0],     # Regularization strength
        'l1_ratio': [0.1, 0.5, 0.9]    # Balance between L1 and L2 penalty
    }


    # Setting up GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)

    # Fit the model
    grid_search.fit(train_data, train_labels)

    # Best parameters from grid search
    best_params = grid_search.best_params_
    print(f"Best parameters: {best_params}")

    # Using best parameters to train the final Elastic Net model
    best_elastic_net = ElasticNet(alpha=best_params['alpha'], l1_ratio=best_params['l1_ratio'])
    best_elastic_net.fit(train_data, train_labels)

    # Make predictions
    y_pred = best_elastic_net.predict(train_data)

    # Evaluate the model
    mse = mean_squared_error(train_labels, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(train_labels, y_pred)
    r2 = r2_score(train_labels, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {rmse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"r2: {r2}")



### RANDOM FOREST REGRESSOR MODEL ###

def rf_regressor():

    # Initialize and train the Random Forest Regressor
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(train_data, train_labels)

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


if __name__ == "__main__":
    # Identify columns with Inf values
    columns_with_inf = train_data.columns[train_data.isin([np.inf, -np.inf]).any(axis=0)]
    print("Columns with Inf values:")
    print(columns_with_inf.tolist())

    # Identify rows with Inf values
    rows_with_inf = train_data.index[train_data.isin([np.inf, -np.inf]).any(axis=1)]
    print("\nRows with Inf values:")
    print(rows_with_inf.tolist())

    # Showing DataFrame elements with Inf values marked
    inf_elements = train_data.applymap(lambda x: "Inf" if np.isinf(x) else x)
    print("\nDataFrame showing Inf values:\n", inf_elements)
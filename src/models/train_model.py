import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

train_data = pd.read_csv('../../data/processed/train_data_processed.csv')
train_labels = pd.read_csv('../../data/processed/train_labels_processed.csv')

train_labels = train_labels['price'] 


### BASELINE LINEAR REGRESSION MODEL ###
def linear_reg_model():
    # Initialize and train the model
    model = LinearRegression()
    model.fit(train_data, train_labels)

    # Make predictions
    y_pred = model.predict(train_data)

    # Evaluate the model
    mse = mean_squared_error(train_labels, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(train_labels, y_pred)
    r2 = r2_score(train_labels, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {rmse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"r2: {r2}")


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

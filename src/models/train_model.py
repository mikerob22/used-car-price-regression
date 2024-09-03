import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

train_data = pd.read_csv('../../data/processed/train_data_processed.csv')
train_labels = pd.read_csv('../../data/processed/train_labels_processed.csv')

train_labels = train_labels['price'] 



def linear_reg_model():
    # Initialize and train the model
    model = LinearRegression()
    model.fit(train_data, train_labels)

    # Make predictions
    y_pred = model.predict(train_data)

    # Evaluate the model
    mse = mean_squared_error(train_labels, y_pred)
    rmse = np.sqrt(mse)

    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {rmse}")



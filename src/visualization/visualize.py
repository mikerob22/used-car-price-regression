from src.data.preprocessing import load_data, handle_missing_values
from src.features.feature_engineering import feature_engineering
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_missing_values(missing_value_counts):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=missing_value_counts.values, y=missing_value_counts.index, palette='viridis')
    
    # Annotate the bars with their values
    for index, value in enumerate(missing_value_counts.values):
        plt.text(value, index, str(value), color='black', ha='left', va='center')
    
    plt.xlabel('Count of Features')
    plt.ylabel('Number of Missing Values per Feature')
    plt.title('Distribution of Missing Values Across Features')
    plt.show()


def scatter_subplots():
    # Create a row of subplots
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))

    # Scatter plot price vs horsepower
    axs[0,0].scatter(feature_engineered['horsepower'], feature_engineered['price'], alpha=0.6, color='blue')
    axs[0,0].set_title('price vs horsepower')
    axs[0,0].set_xlabel('horsepower')
    axs[0,0].set_ylabel('price')

    # Scatter plot price vs engine_size
    axs[0,1].scatter(feature_engineered['engine_size'], feature_engineered['price'], alpha=0.6, color='green')
    axs[0,1].set_title('price vs engine_size')
    axs[0,1].set_xlabel('engine_size')
    axs[0,1].set_ylabel('price')

    # Scatter plot price vs power_to_weight_ratio
    axs[0,2].scatter(feature_engineered['power_to_weight_ratio'], feature_engineered['price'], alpha=0.6, color='red')
    axs[0,2].set_title('price vs power_to_weight_ratio)')
    axs[0,2].set_xlabel('power_to_weight_ratio')
    axs[0,2].set_ylabel('price')

    # Scatter plot in the fourth subplot
    axs[1,0].scatter(feature_engineered['car_age'], feature_engineered['price'], alpha=0.6, color='orange')
    axs[1,0].set_title('price vs car_age')
    axs[1,0].set_xlabel('car_age')
    axs[1,0].set_ylabel('price')

    # Scatter plot in the fifth subplot
    axs[1,1].scatter(feature_engineered['milage'], feature_engineered['price'], alpha=0.6, color='purple')
    axs[1,1].set_title('price vs milage')
    axs[1,1].set_xlabel('milage')
    axs[1,1].set_ylabel('price')

    # Scatter plot in the sixth subplot
    axs[1,2].scatter(feature_engineered['mileage_per_year'], feature_engineered['price'], alpha=0.6, color='brown')
    axs[1,2].set_title('price vs mileage_per_year')
    axs[1,2].set_xlabel('milage_per_year')
    axs[1,2].set_ylabel('price')

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()



def plot_price_by_fuel_type(data):
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Violin plot in first subplot
    sns.violinplot(x='fuel_type', y='price', data=data, ax=axes[0])
    axes[0].set_title('Violin Plot of Price by Fuel Type')
    axes[0].set_ylabel('Price')
    axes[0].set_xlabel('Fuel Type')

    # box plot in second subplot
    data.boxplot(column='price', by='fuel_type', ax=axes[1])
    axes[1].set_title('Box Plot of Price by Fuel Type')
    axes[1].set_ylabel('Price')
    axes[1].set_xlabel('Fuel Type')

    # Display the plots
    plt.tight_layout()
    plt.show()



def price_by_brand(data):
    plt.figure(figsize=(14, 7))
    sns.violinplot(x='brand', y='price', data=data, palette='viridis')
    plt.title('Price Distribution by Brand', fontsize=16)
    plt.xlabel('Brand', fontsize=14)
    plt.ylabel('Price', fontsize=14)
    plt.xticks(rotation=90)
    plt.show()




def numerical_correlations(data):
    corr_matrix = data.corr(numeric_only=True)

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    plt.figure(figsize=(12,6))
    heatmap = sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='0.1g', vmin= -1, vmax= 1, center= 0, cmap= 'rocket', linewidths= 1, linecolor= 'black')
    heatmap.set_title('Correlation HeatMap Between Variables')
    heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=90)



def contingency_tables(data):

    # Bin the 'price' variable into quartiles
    price_bins = pd.qcut(data['price'], q=4, labels=['low', 'medium', 'high', 'very high'])

    # Add the binned price to the dataframe
    data['price_bin'] = price_bins

    # Create contingency tables
    contingency_fuel_type = pd.crosstab(data['fuel_type'], data['price_bin'], margins=True)
    contingency_transmission = pd.crosstab(data['transmission'], data['price_bin'], margins=True)
    contingency_brand = pd.crosstab(data['brand'], data['price_bin'], margins=True)


    # Display the contingency tables
    print("Contingency Table for Fuel Type and Price Bin")
    print(contingency_fuel_type)
    print("\nContingency Table for Transmission and Price Bin")
    print(contingency_transmission)
    print("\nContingency Table for Brand and Price Bin")
    print(contingency_brand)



if __name__ == "__main__":
    data = load_data('data/raw/train.csv')
    missing_value_counts = data.isna().sum()
    plot_missing_values(missing_value_counts)

    data2 = load_data('data/raw/test.csv')
    missing_value_counts2 = data2.isna().sum()
    plot_missing_values(missing_value_counts2)
    
    data['fuel_type'].nunique()

    data_filled = data.fillna({
        'accident': 'Unknown',
        'clean_title': 'Unknown'})

    contingency_accident_title = pd.crosstab(data_filled['accident'], data_filled['clean_title'], margins=True)
import re

# Define regex patterns to extract Horsepower (HP) and engine size (L)
hp_pattern = re.compile(r'(\d+(\.\d+)?)HP')
engine_size_pattern = re.compile(r'(\d+(\.\d+)?)L')


# Function to extract HP
def extract_hp(engine):
    match = hp_pattern.search(engine)
    return float(match.group(1)) if match else None


# Function to extract engine size
def extract_engine_size(engine):
    match = engine_size_pattern.search(engine)
    return float(match.group(1)) if match else None


# Function to add a binary column classifying cars as luxury or not
def luxury_brands_binary(data):
    luxury_brands =  ['Mercedes-Benz', 'BMW', 'Audi', 'Porsche', 'Land', 
                    'Lexus', 'Jaguar', 'Bentley', 'Maserati', 'Lamborghini', 
                    'Rolls-Royce', 'Ferrari', 'McLaren', 'Aston', 'Maybach']

    data['is_luxury_brand'] = data['brand'].apply(lambda x: 1 if x in luxury_brands else 0)
    return None


def feature_engineering(data):

    data['horsepower'] = data['engine'].apply(extract_hp)
    data['engine_size'] = data['engine'].apply(extract_engine_size)
    luxury_brands_binary(data)
    data["car_age"] = 2024 - data["model_year"]
    data['mileage_per_year'] = data['milage'] / data['car_age']
    data['power_to_weight_ratio'] = data['horsepower'] / data['engine_size']

    return data







def feature_engineering(data):
    data["car_age"] = 2024 - data["model_year"]
    return data
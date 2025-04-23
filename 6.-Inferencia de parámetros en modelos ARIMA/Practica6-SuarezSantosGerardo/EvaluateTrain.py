from sklearn.metrics import mean_squared_error
import numpy as np

def evaluate_model(model_fit, data_series, train_size):
    
    train, test = data_series[:train_size], data_series[train_size:]
    
    forecast = model_fit.forecast(steps=len(test))

    mse = mean_squared_error(test, forecast)
    
    rmse = np.sqrt(mse)
    
    return forecast, mse, rmse
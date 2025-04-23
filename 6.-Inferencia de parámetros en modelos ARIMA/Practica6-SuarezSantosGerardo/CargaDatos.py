from datetime import datetime
import pandas as pd

def load_and_prepare_data(file_path, column_name):
    
    data = pd.read_csv(file_path)
    
    data['Date'] = pd.to_datetime(data['Date'])
    
    data.set_index('Date', inplace=True)
    
    data_series = data[column_name]
    
    return data_series

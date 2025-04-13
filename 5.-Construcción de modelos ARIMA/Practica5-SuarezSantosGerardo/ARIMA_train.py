from statsmodels.tsa.arima.model import ARIMA

def train_arima_model(data_series, p, d, q):
    model = ARIMA(data_series, order=(p, d, q))
    model_fit = model.fit()

    return model_fit

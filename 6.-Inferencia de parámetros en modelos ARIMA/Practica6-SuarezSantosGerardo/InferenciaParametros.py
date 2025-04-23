from pmdarima.arima import auto_arima
import warnings

warnings.filterwarnings("ignore")

def infer_parameters(data_series):
    model = auto_arima(data_series,
                       seasonal=False,
                       stepwise=True,
                       trace=True,
                       suppress_warnings=True,
                       error_action='ignore',
                       max_p=5, max_q=5, max_d=2)
    
    print(f"\nMejor modelo ARIMA encontrado: {model.order}\n")
    return model.order

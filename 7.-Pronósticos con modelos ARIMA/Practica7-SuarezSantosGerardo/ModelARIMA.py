from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

class ModelARIMA:
    def __init__(self, time_series):
        self.original_ts = time_series
        self.ts = time_series.copy()
        self.model = None
        self.model_fit = None
        self.order = (0, 0, 0)
        self.diff_order = 0

    def check_stationarity(self):
        result = adfuller(self.ts.dropna())
        print(f"Dickey-Fuller p-value: {result[1]}")
        return result[1] < 0.05  # Si es menor a 0.05, es estacionaria

    def difference_until_stationary(self, max_diff=2):
        d = 0
        ts_diff = self.ts.copy()
        while d <= max_diff:
            if self.check_stationarity():
                self.ts = ts_diff
                self.diff_order = d
                print(f"Serie estacionaria con d = {d}")
                return d
            ts_diff = ts_diff.diff().dropna()
            self.ts = ts_diff
            d += 1
        print("No se logró la estacionariedad con diferenciación limitada.")
        return d
    
    def fit_model(self, p=1, q=1):
        try:
            d = self.diff_order
            print(f"Ajustando modelo ARIMA({p},{d},{q})...")
            self.model = ARIMA(self.original_ts, order=(p, d, q))
            self.model_fit = self.model.fit()
            self.order = (p, d, q)
            print("Modelo ajustado correctamente.")
            return self.model_fit
        except Exception as e:
            print(f"Error al ajustar el modelo: {e}")
            return None
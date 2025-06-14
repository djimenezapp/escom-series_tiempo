import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.filters.hp_filter import hpfilter
from scipy.signal import butter, filtfilt, periodogram
import warnings
warnings.filterwarnings('ignore')

class DataPreparation:
    def __init__(self, filepath):
        self.filepath = filepath
        self.series = None

    def load_and_aggregate(self):
        df = pd.read_csv(self.filepath, parse_dates=['Date'])
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.groupby('Date').size().rename("Accident_Count").reset_index()
        full_range = pd.date_range(start=df['Date'].min(), end=df['Date'].max())
        df = df.set_index('Date').reindex(full_range, fill_value=0).rename_axis('Date').reset_index()
        self.series = df.set_index('Date')['Accident_Count']
        return self.series
    
class ExploratoryAnalysis:
    def __init__(self, series):
        self.series = series

    def plot_series(self):
        self.series.plot(figsize=(12, 4), title="Conteo Diario de Accidentes")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def describe(self):
        return self.series.describe()
    
class StationarityTest:
    def __init__(self, series):
        self.series = series
        self.result = None
        self.diff_series = None
        self.d = 0

    def adf_test(self):
        self.result = adfuller(self.series)
        while self.result[1] > 0.05:
            self.series = self.series.diff().dropna()
            self.result = adfuller(self.series)
            self.d += 1
        self.diff_series = self.series
        return self.diff_series, self.d, self.result[1]
    
class ModelIdentification:
    def __init__(self, series):
        self.series = series

    def plot_acf_pacf(self):
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        plot_acf(self.series, ax=ax[0])
        plot_pacf(self.series, ax=ax[1])
        plt.tight_layout()
        plt.show()

class ModelEstimation:
    def __init__(self, series, order):
        self.series = series
        self.order = order
        self.model = None
        self.result = None

    def fit(self):
        self.model = ARIMA(self.series, order=self.order)
        self.result = self.model.fit()
        return self.result.summary()

    def plot_residuals(self):
        residuals = self.result.resid
        fig, ax = plt.subplots(2, 1, figsize=(10, 6))
        residuals.plot(ax=ax[0], title="Residuos")
        plot_acf(residuals.dropna(), ax=ax[1])
        plt.tight_layout()
        plt.show()

class Forecasting:
    def __init__(self, model_result):
        self.model_result = model_result

    def forecast(self, steps=30):
        forecast = self.model_result.get_forecast(steps=steps)
        pred = forecast.predicted_mean
        conf_int = forecast.conf_int()
        return pred, conf_int
    
class Filters:
    def __init__(self, series):
        self.series = series

    def decompose(self, model='additive'):
        result = seasonal_decompose(self.series, model=model, period=7)
        result.plot()
        plt.tight_layout()
        plt.show()
        return result

    def hp_filter(self, lamb=1600):
        cycle, trend = hpfilter(self.series, lamb=lamb)
        return trend, cycle

    def smoothing(self, window=7):
        return self.series.rolling(window=window).mean()

    def butterworth_filter(self, cutoff, fs, btype='low'):
        b, a = butter(N=4, Wn=cutoff / (0.5 * fs), btype=btype)
        filtered = filtfilt(b, a, self.series)
        return pd.Series(filtered, index=self.series.index)

    def spectral_density(self):
        f, Pxx = periodogram(self.series, scaling='density')
        plt.figure(figsize=(10, 4))
        plt.semilogy(f, Pxx)
        plt.title("Densidad Espectral Potencial")
        plt.xlabel("Frecuencia")
        plt.ylabel("Potencia")
        plt.tight_layout()
        plt.show()
        return f, Pxx
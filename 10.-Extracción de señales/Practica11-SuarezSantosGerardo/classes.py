import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.filters.hp_filter import hpfilter
from scipy.signal import wiener
import matplotlib.pyplot as plt

# Para filtro de Kalman, probar si es que existe en el sistema
try:
    from filterpy.kalman import KalmanFilter
except ImportError:
    KalmanFilter = None
    print("filterpy no instalado, filtro Kalman no disponible.")


class BoxJenkins:
    def __init__(self, series):
        """
        series: pd.Series con índice datetime
        """
        self.series = series
        self.model = None
        self.model_fit = None

    def identificar(self):
        print("Prueba Dickey-Fuller aumentada para estacionariedad:")
        result = adfuller(self.series)
        print(f"ADF Statistic: {result[0]:.4f}")
        print(f"p-value: {result[1]:.4f}")
        for key, val in result[4].items():
            print(f"Critical Value {key}: {val:.4f}")

        print("\nGráficas ACF y PACF:")
        fig, axes = plt.subplots(1, 2, figsize=(14,4))
        plot_acf(self.series, lags=30, ax=axes[0])
        plot_pacf(self.series, lags=30, ax=axes[1])
        plt.show()

    def estimar(self, order):
        """
        Ajusta un modelo ARIMA(p,d,q)
        order: tuple (p,d,q)
        """
        print(f"Ajustando modelo ARIMA{order}...")
        self.model = ARIMA(self.series, order=order)
        self.model_fit = self.model.fit()
        print(self.model_fit.summary())

    def validar(self):
        if self.model_fit is None:
            print("Modelo no ajustado aún")
            return
        resid = self.model_fit.resid
        print("Análisis de residuos:")
        plt.figure(figsize=(10,6))
        plt.subplot(211)
        plt.plot(resid)
        plt.title("Residuos")
        plt.subplot(212)
        plot_acf(resid, lags=30)
        plt.show()

        print(f"Media de residuos: {np.mean(resid):.4f}")
        print(f"Varianza de residuos: {np.var(resid):.4f}")

    def pronosticar(self, pasos=10):
        if self.model_fit is None:
            print("Modelo no ajustado aún")
            return
        forecast = self.model_fit.get_forecast(steps=pasos)
        pred = forecast.predicted_mean
        conf_int = forecast.conf_int()
        return pred, conf_int


class FiltrosSeriesTiempo:
    def __init__(self, series):
        """
        series: pd.Series con índice datetime
        """
        self.series = series

    def filtro_wiener(self):
        filtered = wiener(self.series.values)
        return pd.Series(filtered, index=self.series.index)

    def filtro_ewma(self, span=5):
        filtered = self.series.ewm(span=span, adjust=False).mean()
        return filtered

    def filtro_kalman(self):
        if KalmanFilter is None:
            raise ImportError("filterpy no instalado, no se puede aplicar filtro Kalman.")
        kf = KalmanFilter(dim_x=1, dim_z=1)
        kf.x = np.array([[self.series.iloc[0]]])  # estado inicial
        kf.F = np.array([[1]])  # matriz de transición
        kf.H = np.array([[1]])  # matriz de observación
        kf.P *= 1000.  # incertidumbre
        kf.R = 5  # varianza ruido 
        kf.Q = 0.1  

        estado_estimado = []
        for z in self.series.values:
            kf.predict()
            kf.update(z)
            estado_estimado.append(kf.x[0, 0])
        return pd.Series(estado_estimado, index=self.series.index)

    def filtro_hp(self, lamb=1600):
        ciclo, tendencia = hpfilter(self.series, lamb=lamb)
        return tendencia, ciclo

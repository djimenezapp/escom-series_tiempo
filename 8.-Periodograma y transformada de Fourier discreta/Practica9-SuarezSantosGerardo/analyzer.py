import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings

class TrafficTimeSeriesAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.ts = None
        self.model = None
        self.result = None

    def cargar_datos(self):
        self.df = pd.read_csv(self.file_path, parse_dates=['Date'])
        serie_diaria = self.df.groupby('Date').size()
        self.ts = serie_diaria.asfreq('D').fillna(0)

    def graficar_serie(self):
        self.ts.plot(title='Accidentes por día')
        plt.xlabel('Fecha')
        plt.ylabel('Conteo')
        plt.tight_layout()
        plt.show()

    def analizar_espectro(self):
        n = len(self.ts)
        X = np.fft.fft(self.ts)
        freqs = np.fft.fftfreq(n)
        potencia = np.abs(X)**2
        plt.plot(freqs[:n//2], potencia[:n//2])
        plt.title('Periodograma (Espectro de frecuencia)')
        plt.xlabel('Frecuencia')
        plt.ylabel('Potencia')
        plt.tight_layout()
        plt.show()

    def mostrar_acf_pacf(self, lags=30):
        fig, ax = plt.subplots(2, 1, figsize=(10, 6))
        plot_acf(self.ts, lags=lags, ax=ax[0])
        plot_pacf(self.ts, lags=lags, ax=ax[1])
        plt.tight_layout()
        plt.show()

    def ajustar_modelo(self, orden=(1,1,1)):
        self.model = ARIMA(self.ts, order=orden)
        self.result = self.model.fit()
        print(self.result.summary())

    def diagnosticar_modelo(self):
        residuos = self.result.resid
        plt.plot(residuos)
        plt.title('Residuos del modelo ARIMA')
        plt.tight_layout()
        plt.show()
        plot_acf(residuos, lags=30)
        plt.title('ACF de los residuos')
        plt.tight_layout()
        plt.show()

    def pronosticar(self, pasos=15):
        forecast = self.result.get_forecast(steps=pasos)
        pred_mean = forecast.predicted_mean
        conf_int = forecast.conf_int()
        plt.plot(self.ts, label='Datos reales')
        plt.plot(pred_mean.index, pred_mean, label='Pronóstico', color='green')
        plt.fill_between(conf_int.index, conf_int.iloc[:,0], conf_int.iloc[:,1], color='green', alpha=0.3)
        plt.legend()
        plt.title(f'Pronóstico a {pasos} días')
        plt.tight_layout()
        plt.show()

    def buscar_mejor_modelo_arima(self, max_p=3, max_d=2, max_q=3):
        mejor_aic = float("inf")
        mejor_orden = None
        mejor_modelo = None

        print("Buscando el mejor modelo ARIMA...")
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            modelo = ARIMA(self.ts, order=(p,d,q)).fit()
                        if modelo.aic < mejor_aic:
                            mejor_aic = modelo.aic
                            mejor_orden = (p, d, q)
                            mejor_modelo = modelo
                    except:
                        continue

        if mejor_modelo:
            print(f"Mejor modelo encontrado: ARIMA{mejor_orden} con AIC={mejor_aic:.2f}")
            self.model = ARIMA(self.ts, order=mejor_orden)
            self.result = mejor_modelo
            print(self.result.summary())
        else:
            print("No se encontró un modelo válido.")
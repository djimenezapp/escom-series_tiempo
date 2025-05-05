import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

class Visualizer:
    def __init__(self, time_series):
        self.ts = time_series

    def plot_time_series(self):
        plt.figure(figsize=(14, 5))
        sns.lineplot(data=self.ts)
        plt.title("Accidentes de Tráfico Globales por Día")
        plt.xlabel("Fecha")
        plt.ylabel("Número de Accidentes")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_acf_pacf(self, lags=30):
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        plot_acf(self.ts, ax=axes[0], lags=lags)
        plot_pacf(self.ts, ax=axes[1], lags=lags, method='ywm')
        axes[0].set_title("Autocorrelación (ACF)")
        axes[1].set_title("Autocorrelación Parcial (PACF)")
        plt.tight_layout()
        plt.show()

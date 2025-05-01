import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

if __name__ == "__main__":
    df = pd.read_pickle("df.pkl")
    close_prices = df['Close']
    close_diff = close_prices.diff().dropna()

    # Gráfico de ACF
    plot_acf(close_diff, lags=30)
    plt.title("Función de Autocorrelación (ACF)")
    plt.show()

    # Gráfico de PACF
    plot_pacf(close_diff, lags=30, method='ywm')
    plt.title("Función de Autocorrelación Parcial (PACF)")
    plt.show()
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMAResults

def diagnostico_residuos(residuos):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(residuos, bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_title("Histograma de Residuos")

    axes[1].plot(residuos, color='blue', alpha=0.6)
    axes[1].set_title("Residuos del Modelo")
    plt.show()

    sm.qqplot(residuos, line='s', fit=True)
    plt.title("Q-Q Plot de los Residuos")
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 5))
    sm.graphics.tsa.plot_acf(residuos, lags=30, ax=ax)
    plt.title("ACF de los Residuos")
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 5))
    sm.graphics.tsa.plot_pacf(residuos, lags=30, ax=ax, method='ywm')
    plt.title("PACF de los Residuos")
    plt.show()

if __name__ == "__main__":
    resultado = ARIMAResults.load("modelo_arima.pkl")
    residuos = resultado.resid
    diagnostico_residuos(residuos)

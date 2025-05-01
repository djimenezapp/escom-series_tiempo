import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def entrenar_modelo_arima(serie, orden=(1,1,1)):
    modelo = ARIMA(serie, order=orden)
    resultado = modelo.fit()
    print(resultado.summary())
    resultado.save("modelo_arima.pkl")

if __name__ == "__main__":
    df = pd.read_pickle("df.pkl")
    close_prices = df['Close']
    entrenar_modelo_arima(close_prices, orden=(1, 1, 1))  # parametros del podelo ARIMA

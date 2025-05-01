import pandas as pd
from statsmodels.tsa.stattools import adfuller

def prueba_adf(serie):
    resultado = adfuller(serie.dropna())
    print("ADF Statistic:", resultado[0])
    print("p-value:", resultado[1])
    if resultado[1] < 0.05:
        print("La serie es estacionaria")
    else:
        print("La serie NO es estacionaria")


if __name__ == "__main__":
    df = pd.read_pickle("df.pkl")
    df = df.sort_index()
    close_prices = df['Close']
    prueba_adf(close_prices)
    
"""
Importamos la libreria de statsmodels para hacer uso de la herramienta adfuller, la cual nos ayuda a analizar si la serie es 
estacionaria o no, y usamos numpy para el manejo de valores infinitos
"""
from statsmodels.tsa.stattools import adfuller
import numpy as np

#La clase Prepocesamiento nos ayudará aplicar la prueba de Dickey Fuller y de ser necesario diferenciar la serie de tiempo
class Preprocesamiento:
    #Inicializamos el objeto con el conjunto de datos
    def __init__(self, df):
        self.df = df

    #Ahora lo más importante es ordenar las fechas para posteriores análisis y principalmente para poder aplicar la prueba estadística
    def ordenar_fechas(self):
        #Ordenamos las fechas con .sort_index y retornamos el conjunto de datos ordenados
        self.df.sort_index(inplace = True)
        return self.df
    
    """
    Ahora para aplicar la prueba de Dickey Fuller a nuestra serie de tiempo, como anteriormente mencionamos, hacemos uso del 
    módulo "adfuller" de statsmodels, tenemos para esta prueba dos hipótesis:
                                            H0 = La serie no es estacionaria
                                            H1 < 0 = la serie es estacionaria
    La prueba nos dirá mediante un estadístico de prueba, un p-valor, y un valor crítico si la serie es estacionaria, si el p-valor 
    es mayor a 0.05 (un 95% de confianza) significará que la serie no es estacionaria (Fracaso para rechazar H0) y tendrá que pasar
    por una transformación para que esta sea estacionaria (Rechazamos H0).
    """
    def prueba_dickey_fuller(self, columna):
        serie = self.df[columna]
        serie = serie.replace([np.inf, -np.inf], np.nan)  # Reemplazamos los valores infinitos por NaN
        serie = serie.dropna()  # Eliminamos los valores nulos
        resultado = adfuller(serie)
        return {
            'Estadístico de Prueba': resultado[0],
            'P-valor': resultado[1],
            'Valor Critico': resultado[4]
        }
    
    """
    La diferenciación de la serie nos ayuda a hacer que la serie se vuelva estacionaria en el caso de que la prueba de Dickey
    Fuller nos de que la sere no es estacionaria, la serie puede pasar por múltiples diferenciaciones pero lo normal es que en 
    dos diferencias la serie se vuelva estacionaria. 
    """    
    def diferenciar(self, columna):
        serie_diferenciada = self.df[columna].diff().dropna()
        return serie_diferenciada

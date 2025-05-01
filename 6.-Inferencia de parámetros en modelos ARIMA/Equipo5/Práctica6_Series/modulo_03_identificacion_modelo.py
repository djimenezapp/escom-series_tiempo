#Importamos la libreria matplotlib para poder realizar la parte gráfica y statsmodels para calculas las funciones de autocorrelación
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf, pacf

class IdentificacionModelo:

    def __init__(self, serie):
        self.serie = serie

    def graficar_fac_facp(self, retrasos=5):
        fig, axes = plt.subplots(1, 2, figsize = (20,10))
        sm.graphics.tsa.plot_acf(self.serie, lags = retrasos, ax=axes[0])
        sm.graphics.tsa.plot_pacf(self.serie, lags= retrasos, ax=axes[1])
        axes[0].set_title("Función de Autocorrelación (FAC) de la serie")
        axes[1].set_title("Función de Autocorrelación parcial (FACP) de la serie")
        plt.show()
    
    def calcular_fac_facp(self, retrasos = 5):
        fac = acf(self.serie.dropna(), nlags = retrasos)
        facp = pacf(self.serie.dropna(), nlags = retrasos)
        return fac, facp

"""
Algo muy importante de este módulo, es que no te da los parámetros del modelo, a diferencia si usáramos auto_arima el cual ya nos
da el valor de los parámetros de este modelo, aquí entra la parte analítica de la persona para identificar los mejores valores
identificando patrones o casos particulares de cada una de las funciones:

En la FAC podemos analizar el posible valor de "q" (componente de media móvil (MA) de ARIMA), si los valores de las autocorrelaciones
se corta de manera abrupta, entonces el valor donde corta la gráfica sera el valor de "q"

En la FACP se analiza el posible valor de "p" (componente autorregresivo (AR) de ARIMA), igual que en la FAC, si los valores de 
las autocorrelaciones se corta de manera abrupta, entonces el valor donde corta la gráfica sera el valor de "p"
"""
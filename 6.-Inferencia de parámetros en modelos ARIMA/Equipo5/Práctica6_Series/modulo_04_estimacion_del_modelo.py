#Importamos la libreria statsmodels para hacer uso de la función ARIMA que nos devuelve el modelo del mismo nombre
from statsmodels.tsa.arima.model import ARIMA

#La clase EstimarModelo nos ayuda en la creación de nuestro modelo ARIMA con el orden que se haya identificado
class EstimarModelo:
    
    #Inicalizamos el objeto con la serie de tiempo y el orden que va a tener el ARIMA
    def __init__(self, serie, orden):
        self.serie = serie

        """
        La variable orden debera contener el valor del componente p (autorregresivo), de la cantidad de diferenciaciones que se 
        realizaron a la serie (componente d), y el valor del componente q (promedio móvil) en ese orden
        """
        self.orden = orden #p,d,q

    #Creamos el modelo con la serie de tiempo y parámetros establecidos en la variable orden
    def entrenar(self):
        modelo = ARIMA(self.serie, order = self.orden)
        
        #Ajustamos el modelo
        modelo_ajuste = modelo.fit()
        
        #Devolvemos el modelo ajustado
        return modelo_ajuste
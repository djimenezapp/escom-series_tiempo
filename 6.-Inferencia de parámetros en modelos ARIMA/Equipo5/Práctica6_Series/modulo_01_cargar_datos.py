#Importamos pandas para poder leer el conjunto de datos en formato .csv
import pandas as pd

#La clase cargar datos nos permitira leer nuestro dataset para posteriormente devolverlo y asignarlo a una variable
class CargarDatos:

    #Inicializamos el objeto con la ruta en la que esta el .csv
    def __init__(self, ruta, columna_fecha):
        self.columna_fecha = columna_fecha
        self.ruta = ruta
    
    """
    Leemos el conjunto de datos con la función .read_csv de pandas, y le pasamos como parámetros que queremos convertir la columna
    de fechas en indice y se trata de llevar a esta columna a un formato tradicional y común
    """
    def cargar(self):
        df = pd.read_csv(self.ruta, parse_dates=[self.columna_fecha], index_col=self.columna_fecha)
        print("Datos cargados exitosamente...")
        return df

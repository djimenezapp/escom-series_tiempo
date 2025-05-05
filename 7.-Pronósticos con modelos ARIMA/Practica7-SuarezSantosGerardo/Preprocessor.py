import pandas as pd

class Preprocessor:
    def __init__(self, dataframe):
        self.df = dataframe.copy()
        self.time_series = None

    def preprocess(self):
        try:
            # Convertir columna 'Date' a datetime
            self.df['Date'] = pd.to_datetime(self.df['Date'], errors='coerce')
            self.df = self.df.dropna(subset=['Date'])

            # Agrupar por fecha y contar el número de accidentes
            daily_counts = self.df.groupby('Date').size().rename("Accidents")

            # Asegurar que sea una serie de tiempo con índice de fechas
            daily_counts = daily_counts.asfreq('D').fillna(0)

            self.time_series = daily_counts
            print("Serie temporal preparada correctamente.")
            return self.time_series
        except Exception as e:
            print(f"Error en el preprocesamiento: {e}")
            return None

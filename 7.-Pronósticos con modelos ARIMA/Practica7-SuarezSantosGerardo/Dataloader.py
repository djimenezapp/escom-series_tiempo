import pandas as pd

file_path = "global_traffic_accidents.csv"

class DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None

    def load_data(self):
        try:
            df = pd.read_csv(self.filepath)
            required_columns = ['Date', 'Accident ID']
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Falta la columna requerida: {col}")
            self.data = df
            print("Datos cargados correctamente.")
            return self.data
        except Exception as e:
            print(f"Error al cargar los datos: {e}")
            return None

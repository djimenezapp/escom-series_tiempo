import pandas as pd
import sys

def cargar_datos(ruta_csv, columna_fecha="Date"):
    df = pd.read_csv(ruta_csv, parse_dates=[columna_fecha])#se entiende que son series de tiempo
    df.set_index(columna_fecha, inplace=True)
    return df

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("python 01_cargar_datos.py <ruta_csv>")
        sys.exit(1)
    
    ruta_csv = sys.argv[1]
    df = cargar_datos(ruta_csv)
    df.to_pickle("df.pkl")
    print("Datos cargados y guardados como df.pkl")
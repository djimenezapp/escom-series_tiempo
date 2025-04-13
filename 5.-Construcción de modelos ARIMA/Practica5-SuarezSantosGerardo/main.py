import CargaDatos
import AC_analysis
import ARIMA_train
import EvaluateTrain
import pandas as pd

def main():

    file_path = 'F:/ESCOM-activo/Series de Tiempo/Practica5-SuarezSantosGerardo/global_traffic_accidents.csv' #Cambia el nombre del archivo

    temp = pd.read_csv(file_path)

    print("\n Columnas del archivo seleccionado: \n")

    print(temp.columns)

    column_name = input("\n Ingrese el nombre de la columna para el análisis temporal: \n")

    data_series = CargaDatos.load_and_prepare_data(file_path, column_name)

    AC_analysis.plot_acf_pacf(data_series)

    model_fit = ARIMA_train.train_arima_model(data_series, p=1, d=1, q=1)

    train_size = int(len(data_series) * 0.8) #80 entrenamiento, 20 prueba
    forecast, mse, rmse = EvaluateTrain.evaluate_model(model_fit, data_series, train_size)

    #Resultados:
    print("Primeros 10 pronósticos:", forecast[:10]) #Ajustado a 10 por defecto
    print("Error cuadrático medio (MSE):", mse)
    print("Raíz del error cuadrático medio (RMSE):", rmse)

if __name__ == "__main__":
    main()
import pandas as pd
import matplotlib.pyplot as plt
from classes import BoxJenkins
from classes import FiltrosSeriesTiempo

def main():

    print("Cargando dataset...")
    df = pd.read_csv('Practica11-SuarezSantosGerardo\global_traffic_accidents.csv', parse_dates=['Date'])
    df_agg = df.groupby('Date')['Casualties'].sum()
    

    print("Inicializando análisis Box-Jenkins...")
    bj = BoxJenkins(df_agg)
    filtros = FiltrosSeriesTiempo(df_agg)
    

    print("\n--- Fase de Identificación ---")
    bj.identificar()
    

    print("\n--- Fase de Estimación ---")
    bj.estimar(order=(1,1,1))
    

    print("\n--- Fase de Validación ---")
    bj.validar()
    

    print("\n--- Fase de Pronóstico ---")
    pred, conf_int = bj.pronosticar(pasos=15)
    print(pred)
    

    print("\nAplicando filtros...")
    wiener_filt = filtros.filtro_wiener()
    ewma_filt = filtros.filtro_ewma(span=7)
    try:
        kalman_filt = filtros.filtro_kalman()
    except ImportError:
        kalman_filt = None
        print("Filtro Kalman no disponible.")
    hp_tendencia, hp_ciclo = filtros.filtro_hp()
    

    plt.figure(figsize=(14,7))
    plt.plot(df_agg, label='Original', color='black')
    plt.plot(wiener_filt, label='Filtro Wiener', linestyle='--')
    plt.plot(ewma_filt, label='Filtro EWMA', linestyle='--')
    if kalman_filt is not None:
        plt.plot(kalman_filt, label='Filtro Kalman', linestyle='--')
    plt.plot(hp_tendencia, label='Filtro Hodrick-Prescott (Tendencia)', linestyle='--')
    plt.title("Comparación de filtros en la serie de víctimas por accidentes")
    plt.xlabel("Fecha")
    plt.ylabel("Número de víctimas")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
from analyzer import TrafficTimeSeriesAnalyzer

def main():
    archivo = "Practica9-SuarezSantosGerardo\global_traffic_accidents.csv"

    # Crear y usar el analizador
    analizador = TrafficTimeSeriesAnalyzer(archivo)

    analizador.cargar_datos()
    analizador.graficar_serie()
    analizador.analizar_espectro()
    analizador.mostrar_acf_pacf()
    #analizador.ajustar_modelo(orden=(1,1,1)) 
    analizador.buscar_mejor_modelo_arima(max_p=3, max_d=3, max_q=3)
    analizador.diagnosticar_modelo()
    analizador.pronosticar(pasos=15)

if __name__ == "__main__":
    main()
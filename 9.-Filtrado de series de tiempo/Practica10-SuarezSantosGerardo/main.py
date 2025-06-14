import matplotlib.pyplot as plt
from analyzer import (           
    DataPreparation,
    ExploratoryAnalysis,
    StationarityTest,
    ModelIdentification,
    ModelEstimation,
    Forecasting,
    Filters,
)

def main() -> None:
    FILE = "Practica10-SuarezSantosGerardo\global_traffic_accidents.csv"
    prep = DataPreparation(FILE)
    series = prep.load_and_aggregate()
    exp = ExploratoryAnalysis(series)
    exp.plot_series()           
    print(exp.describe())   

    stat = StationarityTest(series)
    diff_series, d, p_value = stat.adf_test()
    print(f"Diferenciaciones requeridas: {d}  |  p‑value ADF final: {p_value:.4g}")

    ident = ModelIdentification(diff_series)
    ident.plot_acf_pacf()      
    p, q = 1, 1      

    est = ModelEstimation(series, order=(p, d, q))
    print(est.fit())            
    est.plot_residuals() 

    forecaster = Forecasting(est.result)
    pred, ci = forecaster.forecast(steps=30)   

    plt.figure(figsize=(10, 4))
    series[-90:].plot(label="Observado")
    pred.plot(label="Pronóstico")
    plt.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1], alpha=0.25)
    plt.title("Pronóstico ARIMA")
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

    flt = Filters(series)

    # 7.1 Descomposición aditiva semanal
    decomp = flt.decompose(model="additive")

    # 7.2 Filtro Hodrick‑Prescott
    trend, cycle = flt.hp_filter()
    trend.plot(figsize=(10, 3), title="Tendencia HP"); plt.grid(True); plt.tight_layout(); plt.show()
    cycle.plot(figsize=(10, 3), title="Ciclo HP"); plt.grid(True); plt.tight_layout(); plt.show()

    # 7.3 Suavizamiento por media móvil de 7 días
    smooth = flt.smoothing(window=7)
    smooth.plot(figsize=(10, 3), title="Suavizamiento 7 días"); plt.grid(True); plt.tight_layout(); plt.show()

    # 7.4 Filtros pasa bajas / pasa altas Butterworth
    low_pass  = flt.butterworth_filter(cutoff=0.05, fs=1, btype="low")
    high_pass = flt.butterworth_filter(cutoff=0.05, fs=1, btype="high")
    plt.figure(figsize=(10, 3))
    low_pass.plot(label="Pasa bajas")
    high_pass.plot(label="Pasa altas")
    plt.title("Butterworth: bajas vs altas"); plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

    # 7.5 Densidad Espectral Potencial
    flt.spectral_density()

if __name__ == "__main__":
    main()
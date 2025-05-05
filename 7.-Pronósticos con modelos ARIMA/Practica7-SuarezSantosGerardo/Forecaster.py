import matplotlib.pyplot as plt

class Forecaster:
    def __init__(self, model_fit, original_series):
        self.model_fit = model_fit
        self.original_series = original_series

    def forecast(self, steps=30):
        try:
            forecast_result = self.model_fit.get_forecast(steps=steps)
            forecast_mean = forecast_result.predicted_mean
            conf_int = forecast_result.conf_int()

            # Crear gráfico
            plt.figure(figsize=(14, 6))
            plt.plot(self.original_series.index, self.original_series, label="Histórico")
            plt.plot(forecast_mean.index, forecast_mean, color="orange", label="Pronóstico")
            plt.fill_between(
                forecast_mean.index,
                conf_int.iloc[:, 0],
                conf_int.iloc[:, 1],
                color='orange',
                alpha=0.3,
                label="95% IC"
            )
            plt.title(f"Pronóstico de Accidentes (Próximos {steps} días)")
            plt.xlabel("Fecha")
            plt.ylabel("Accidentes diarios")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

            return forecast_mean, conf_int
        except Exception as e:
            print(f"Error en el pronóstico: {e}")
            return None, None
from Dataloader import DataLoader
from Preprocessor import Preprocessor
from Visualizer import Visualizer
from ModelARIMA import ModelARIMA
from Forecaster import Forecaster

class MainARIMAAnalysis:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None
        self.time_series = None
        self.model_fit = None

    def run(self):
        print("\n---INICIO DEL ANÁLISIS DE SERIES DE TIEMPO ---\n")

        loader = DataLoader(self.filepath)
        self.data = loader.load_data()
        if self.data is None:
            return

        preprocessor = Preprocessor(self.data)
        self.time_series = preprocessor.preprocess()
        if self.time_series is None:
            return

        visualizer = Visualizer(self.time_series)
        visualizer.plot_time_series()
        visualizer.plot_acf_pacf()

        modeler = ModelARIMA(self.time_series)
        modeler.difference_until_stationary()
        self.model_fit = modeler.fit_model(p=2, q=2)
        if self.model_fit is None:
            return

        print(self.model_fit.summary())

        forecaster = Forecaster(self.model_fit, self.time_series)
        forecast_values, _ = forecaster.forecast(steps=30)

# Ejecutar
if __name__ == "__main__":
    analysis = MainARIMAAnalysis("global_traffic_accidents.csv")
    analysis.run()
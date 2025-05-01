#Importamos la libreria matplotlib para la parte gráfica y statsmodels para la evaluación del modelo
import matplotlib.pyplot as plt
import statsmodels.api as sm

#La clase EvaluarModelo nos ayudará a analizar los residuos del modelo y ver que tan usable podría llegar a ser este
class EvaluarModelo:
    
    #Inicializamos el objeto con el modelo ajustado
    def __init__(self, modelo_fit):
        self.modelo_fit = modelo_fit
    
    #Para conseguir los residuos del modelo, basta con usar la función .resid de statsmodels y finalmente los graficamos
    def graficar_residuos(self):
        residuos = self.modelo_fit.resid
        fig, ax = plt.subplots(1,2, figsize=(12,6))
        residuos.plot(title="Residuos", ax=ax[0])
        sm.qqplot(residuos, line='s', ax=ax[1])
        ax[1].set_title("Gráfico Q-Q de Residuos")
        ax[0].set_xlabel("Fecha")
        ax[1].set_xlabel("Cuartiles")
        plt.show()

    #Imprimimos el resumen del modelo para ver que tan bien ajustado esta
    def resumen_modelo(self):
        print(self.modelo_fit.summary())
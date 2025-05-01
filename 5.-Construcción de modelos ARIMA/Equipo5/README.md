# Proyecto de Series de Tiempo con ARIMA

Este proyecto implementa un flujo modular como POO para analizar series temporales usando el modelo ARIMA, siguiendo la metodología Box-Jenkins. Cada etapa del análisis está separada en un archivo `.py` para facilitar el mantenimiento y reutilización con distintos datasets.

```
Series_Tiempo/
01_cargar_datos.py           # Cargar y guardar el DataFrame
02_preprocesamiento.py       # Ordenar fechas, prueba Dickey-Fuller
03_modelo_auto_arima.py      # Auto selección de parámetros ARIMA
04_modelo_arima.py           # Entrenamiento del modelo ARIMA
05_diagnostico_modelo.py     # Evaluación de residuos del modelo

df.pkl                       # Archivo generado (serie temporal en pickle)

coin_Bitcoin.csv             # Dataset original
```

### 1. Cargar los datos
```bash
python 01_cargar_datos.py coin_Bitcoin.csv
```
Convierte el CSV en un archivo `df.pkl` con la columna de fecha como índice.

---

### 2. Preprocesamiento
```bash
python 02_preprocesamiento.py
```
Ordena las fechas y aplica la prueba de Dickey-Fuller (ADF) para verificar la estacionariedad de la serie.

---

### 3. Modelo Auto ARIMA
```bash
python 03_modelo_auto_arima.py
```
Grafica la función de autocorrelación (ACF) y autocorrelación parcial (PACF) de la serie diferenciada para ayudar a elegir los valores óptimos de p y q.
---

### 4. Entrenar modelo ARIMA
```bash
python 04_modelo_arima.py
```
Entrena un modelo ARIMA con los valores (p, d, q) elegidos previamente.

---

### 5. Diagnóstico de residuos
```bash
python 05_diagnostico_modelo.py
```
Evalúa los residuos del modelo con histogramas, Q-Q plot, ACF y PACF.
---

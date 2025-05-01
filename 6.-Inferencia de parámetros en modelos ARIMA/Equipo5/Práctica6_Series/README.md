# Práctica de Series de Tiempo para el Análisis de Casos de COVID-19 en México

"""Integrantes del equipo
Diaz Contreras Nuñez Rafael,
Escudero Gutiérrez Evelyn Abril,
Mondragon Aguilar Victor Hugo,
Ramírez Montiel Alejandro
"""


Este proyecto realiza un análisis de series de tiempo aplicando la metodología Box-Jenkins sobre el número de casos confirmados de COVID-19 en México.
Se implementa la identificación, estimación y evaluación de un modelo ARIMA, respetando todas las etapas del proceso, sin utilizar herramientas automáticas como auto_arima.

## Estructura del Proyecto

El proyecto está modularizado en seis partes principales y un archivo main.py:

1.  **`main.py`**: Script principal que orquesta la ejecución de los diferentes Modulo.

2.  **`modulo_01_cargar_datos.py`**: Carga del conjunto de datos CSV, convirtiendo la columna de fechas en índice.

3.  **`modulo_02_preprocesamiento.py`**: Modulo de preprocesamiento de la serie de tiempo, ordena las fechas y la aplicación de pruebas de estacionariedad (Dickey-Fuller) y diferenciación de la serie si es necesario.

4.  **`modulo_03_identificacion_modelo.py`**: Modulo para la identificación visual y analítica del orden del modelo ARIMA, a través de las funciones de autocorrelación (FAC) y autocorrelación parcial (FACP).

5.  **`modulo_04_estimacion_del_modelo.py`**: Modulo que implementa la estimación del modelo ARIMA con los parámetros identificados.

6.  **`modulo_05_evaluacion.py`**:	Evaluación del modelo mediante análisis gráfico de residuos y resumen un estadístico del modelo.

## Flujo de Trabajo

El script `main.py` sigue los siguientes pasos:

1.  **Carga de Datos**:
    * Utiliza la clase `CargarDatos` del módulo `modulo_01_cargar_datos.py` para cargar el archivo `covid.csv`.
    * Filtra los datos para obtener únicamente los registros correspondientes a México.
    * Selecciona las columnas relevantes: 'Country/Region' y 'Confirmed'.

2.  **Preprocesamiento**:
    * Utiliza la clase `Preprocesamiento` del módulo `modulo_02_preprocesamiento.py`.
    * Ordena la serie de tiempo por fecha utilizando el método `ordenar_fechas()`.
    * Realiza pruebas de estacionariedad utilizando la prueba de Dickey-Fuller (`prueba_dickey_fuller()`).
    * Si la serie no es estacionaria, la diferencia utilizando el método `diferenciar()` hasta que la prueba de Dickey-Fuller indique estacionariedad. Se guarda el número de diferenciaciones realizadas.

3.  **Identificación del Modelo**:
    * Utiliza la clase `IdentificacionModelo` del módulo `modulo_03_identificacion_modelo.py`.
    * Calcula y muestra los valores de la Función de Autocorrelación (FAC) y la Función de Autocorrelación Parcial (FACP) para un número de retrasos especificado.
    * Grafica la FAC y la FACP para ayudar en la identificación visual de los órdenes \(p\) y \(q\) del modelo ARIMA.

4.  **Estimación del Modelo**:
    * Utiliza la clase `EstimarModelo` del módulo `modulo_04_estimacion_del_modelo.py`.
    * Crea un modelo ARIMA utilizando la serie de tiempo preprocesada y el orden \((p, d, q)\) identificado (en este caso, \(p=1\), \(d\) es el número de diferenciaciones encontradas, y \(q=0\)).
    * Entrena el modelo con los datos utilizando el método `entrenar()`.

5.  **Evaluación del Modelo**:
    * Utiliza la clase `EvaluarModelo` del módulo `modulo_05_evaluacion.py`.
    * Grafica los residuos del modelo para analizar su comportamiento.
    * Genera un gráfico Q-Q de los residuos para evaluar si siguen una distribución normal.
    * Imprime un resumen del modelo, incluyendo los coeficientes y sus p-valores, así como los criterios de información AIC y BIC.

## Requisitos

Para ejecutar este proyecto, necesitas tener instaladas las siguientes librerías de Python:

* `pandas`
* `statsmodels`
* `numpy`
* `matplotlib`

Puedes instalar estas librerías utilizando pip:

```bash
pip install pandas statsmodels numpy matplotlib

from modulo_01_cargar_datos import CargarDatos
from modulo_02_preprocesamiento import Preprocesamiento
from modulo_03_identificacion_modelo import IdentificacionModelo
from modulo_04_estimacion_del_modelo import EstimarModelo
from modulo_05_evaluacion import EvaluarModelo

# 1. Cargamos los datos

#Pasamos la ruta del conjunto de datos al modulo
cargar = CargarDatos('./covid.csv', 'Date')
df_covid = cargar.cargar()
#print(df_covid.head(10)) #Veemos que se haya cargado el dataset

#Veemos un aproximado de cuantos valores hay por país
conteo = df_covid['Country/Region'].value_counts()
print(conteo)

#Separamos los casos de México
df_covid_mexico = df_covid.loc[df_covid['Country/Region'] == 'Mexico']
#print(df_covid_mexico.head(10)) #Vemos que se haya hecho bien el filtrado

"""
El conjunto de datos tiene la columna de Pais, Provincia, Casos Confirmados, Personas Recuperadas y Muertes, para efectos de esta
práctica solo nos quedaremos con las columnas de Estado y Casos Confirmados (Confirmed)
"""

columnas = ['Country/Region', 'Confirmed']
df_covid_mexico = df_covid_mexico[columnas]
#print(df_covid_mexico.head()) #Checamos que hayamos hecho bien el filtrado

# 2. Preprocesamiento

pre = Preprocesamiento(df_covid_mexico)
df_covid_mexico = pre.ordenar_fechas() #Ordenamos las fechas

bandera = 0 #La bandera nos ayudará para salir del bucle si se cumple una condición
diferenciacion = 0 #Guarda el numero de diferenciaciones que necesitó la serie, esto para usarlo al momento de construir el modelo ARIMA

columna_actual = 'Confirmed' #Fijamos en la columna "Confirmed" para la primera iteración del bucle

while(bandera == 0): #Mientras no se cumpla la condición que esta a continuación, el bucle sigue

    prueba_adf = pre.prueba_dickey_fuller(columna=columna_actual) #Se aplica la prueba de estacionalidad
    print("\n-----------------------------------------------------------------------------------------------------------------------")
    print("Dickey-Fuller: ", prueba_adf)
    print("-----------------------------------------------------------------------------------------------------------------------")

    #Si los resultados de la prueba dan que la serie no es estacionaria, se diferencia hasta que se vuelva estacionaria
    if prueba_adf['P-valor'] > 0.05: 
        nueva_columna = 'Confirmed_diff'
        df_covid_mexico[nueva_columna] = pre.diferenciar(columna_actual)  # Hacer estacionaria si no lo es
        columna_actual = nueva_columna #Cambiamos la columna por la nueva que se creo
        diferenciacion += 1 #Aumentamos en uno el número de diferenciaciones

    #Si se cumple que la prueba da que la serie es estacionaria, se rompe el ciclo
    if prueba_adf['P-valor'] < 0.05:
        print("Número de diferenciaciones: ", diferenciacion)
        bandera = 1
    

# 3. Identificación

"""
Para esta sección, es necesario analizar las gráficas y los valores de la FAC y FACP para inferir que tipo de modelo vamos a usar y
el valor de los parámetros de este.
"""


ident = IdentificacionModelo(df_covid_mexico['Confirmed'])

fac, facp = ident.calcular_fac_facp(retrasos=30) #Calculamos los valores de la FAC y FACP para 30 retrasos

print("\n" + "="*100)
print("Valores FAC")
for i in range(len(fac)):
    print(f"Retraso {i}: {fac[i]}")
print("="*100)

print("\n" + "="*100)
print("Valores FACP")
for i in range(len(facp)):
    print(f"Retraso {i}: {facp[i]}")
print("="*100)

#Se tiene que observar y decidir los valores de p, d y q en ese orden
ident.graficar_fac_facp(retrasos=30)

"""
Analizando la gráfica y la tabla de las FAC y FACP podemos observar que:

    - En la FAC con 30 retrasos, observamos una disminución lenta de los valores, y no hay algun corte abrupto en estos, lo cual
    nos indica un comportamiento autoregresivo típico, por lo que hasta ahorita, no hay algo claro sobre el componente de 
    promedio móviles MA "q", por lo que, ahora deberemos de inferir el valor del componente autoregresivo AR "p", y nos apoyaremos
    de la FACP.

    - Por el lado de la FACP se tiene que después del tiempo 1 los valores bajan abruptamente tendiendo a cero, por lo que 
    podemos decir que 1 retraso en los valores es significativo, dicho en otras palabras, el valor actual (presente) esta 
    influenciado en un 99% (valor dado por la tabla) por el valor pasado, por lo que nos indica que el valor del componente AR
    es 1.

Así que una vez teniendo en cuenta esto, los parámetros que podemos inferir que tiene el modelo son los siguientes:

    - p = 1: Esto porque vemos en la FACP que el valor que más influye en el valor actual, es el pasado inmediato.
    - q = 0: Esto porque en la FAC vemos que los valores disminuyen gradualmente, por lo que la serie tiene un comportamiento más
             autoregresivo que de medias móviles, por lo que podríamos jugar más con el valor de "p" que de "q".
    - d = 2: Por último y no menos importante, la "d" es el número de diferenciaciones por la que debe pasar la serie para ser esta-
             cionaria, este valor lo obtuvimos al final de estar haciendo las pruebas de Dickey Fuller .
"""

# 4. Estimación del modelo

#Recordemos que la variable orden recibe los parámetros en este orden (p,d,q)
modelo = EstimarModelo(df_covid_mexico['Confirmed'], orden=(1,diferenciacion,0)) #Construimos el modelo

#Entrenamos el modelo
modelo_ajustado = modelo.entrenar()

# 5. Evaluación del modelo

diag = EvaluarModelo(modelo_ajustado) #Creamos nuestro objeto para evaluar el modelo

#Hacemos dos gráficos, uno que muestre la variación de los residuos, y otro que nos muestre la distribución de los datos
diag.graficar_residuos()

#Imprimimos el resumen del modelo para analizar los coeficientes del modelo
diag.resumen_modelo()

"""
Analizando las dos gráficas generadas tenemos lo siguiente:

    1. En la gráfica de residuos tenemos que en los primeros meses, la variabilidad de los erros es poca mientras que en los últimos
       meses esta variación aumenta, lo cual hace sentido ya que en los primeros meses de la pandemia en México, el número de casos 
       eran pocos, pero con el tiempo estos aumentaron de forma abrupta y acelerada, por lo que el error de las predicciones se 
       espera que sea grande, ya que el modelo no captura los fenómenos que estan detrás de este crecimiento.
    
    2. En la gráfica de Q-Q tenemos que si los valores siguen una distribución normal teórica, se ajustaran a una linea recta (la
       cual se ve en la misma gráfica), y vemos que más o menos los datos se ajustan a esta linea, ya que la variación de los datos
       es muy poca en los cuartiles -2 a 2, apartir de estos, los valores empiezan a dispersarse mucho, por lo que podríamos decir
       que son datos atípicos, caso que no lo son, ya que aunque los valores son muy altos y no siguen el comportamiento "normal"
       de los datos, son datos reales y casos confirmados que si pasaron, estos fueron debidos a varias circunstancias como el 
       pánico entre las personas, el que no se cuidará la gente u otras eventos que no se previeron o que no se pueden preveer.

Por último tenemos el resumen del modelo, en el cual los valores que más nos interesan son los P-valores de los coeficientes del
modelo, esto ya que si estos valores son menores a 0.05 (caso que si lo es), nos quiere decir que son significativos para el modelo, 
osea que al parecer si tiene un buen ajuste el modelo con esas variables, por otro lado el AIC y el BIC no nos aportan gran 
información al menos que se compararán con otros modelos, ya que al comparar modelos con diferentes parámetros podríamos obtener
un AIC y/o BIC más bajo o alto, pero en general valores bajos de estas dos medidas, nos quieren decir que el modelo es bueno.
"""

# 6. Predicciones
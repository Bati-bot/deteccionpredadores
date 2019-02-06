# Detección de Depredadores Sexuales en salas de chat
Trabajo integrador en el marco del curso Data Science de Digital House - por Juan Ferraro, Gabriel Casal y Santiago Paz

## Descripción General
*"Un depredador sexual es una persona que obtiene o trata de obtener contacto sexual con otra de manera abusiva. De forma análoga a cómo un depredador caza su presa, se cree que el depredador sexual "caza" a sus parejas sexuales."* (Wikipedia en inglés).

El propósito de este trabajo es la identificación de potenciales depredadores sexuales, según la definición arriba citada, a partir de características léxicas o de comportamiento discursivo. 

### Data
La base fue armada por PAN, una organización que fomenta la investigación forense de textos digitales mediante la organización de competencias. Este dataset formó parte de una competencia en 2012. El mismo cuenta con una base de 200.000 conversaciones que comprenden un total de 2.000.000 de registros (lineas de chat) únicos. Las conversaciones de los depredadores sexuales se obtuvieron del sitio Peverted Justice en el que voluntarios se hacen pasar por menores de edad para atrapar a los depredadores; mientras que las conversaciones regulares provienen de distintos sitios de chat. Para asemejarlo a la realidad, se balancearon las conversaciones de modo que sólo el 3% de las mismas correspondan a conversaciones con depredadores.

El evento constaba de dos competencias: la primera buscaba identificar conversaciones en las que participaran depredadores y la segunda  encontrar las líneas de los mismos. Nosotros optamos por esta última, por la mayor dificultad que representaba. Para este tipo de desafíos tan desbalanceados, medidas como el Accuracy no son efectivas, por lo que deben buscarse otras métricas como el area bajo la curva ROC (AUC). En este caso, se tomó como métrica el F3, medida que privilegia el Recall sobre la Precision tratando de identificar el mayor número de depredadores aunque eso signifique clasificar así a algunos inocentes, por lo que será el F3 lo que trataremos de optimizar.

La base venía dividida en un set de train con el 30% de las conversaciones y uno de test con el 70% de conversaciones. No encontramos justificación para esta partición por lo que decidimos invertir los sets. Con esto pretendimos que el modelo tenga más datos para entrenar a la vez que quedaba un test lo suficientemente grande para probar la capacidad predictiva del algoritmo.

Para más detalles sobre la competencia y descargar los datos en formato XML:

https://pan.webis.de/clef12/pan12-web/author-identification.html

http://www.perverted-justice.com/

## Herramientas utilizadas
Se trabajó en Python en un entorno de Jupyter Notebook. Las librerías utilizadas fueron: 
- NumPy
- Pandas
- Scikit-Learn
- Joblib
- TextBlob
- ElementTree
- ScyPy
- Seaborn
- Matplotlib
- ImbLearn. 

# El Trabajo
## 1. Normalizado y agregado de labels
En primer lugar se pasó a formato de tabla la información del XML, donde cada fila de la matriz contenía el texto de una línea de chat, así como información de la misma: hora, autor (hasheado), número de conversación, número de línea dentro de la conversación. Esto se hizo tanto para el set de train como el de test. Luego en se agregó una label de 1 o 0 en función si la línea provenía de un autor clasificado como depredador o no, respectivamente.

## 2. Armado de Features
Para esto se utilizaron 3 enfoques distintos:

### 2.1. Bag of Words y Tf-idf
Convertimos a una matriz modelable todas las palabras de cada línea mediante estos dos métodos. Haciendo una prueba de hiperparámetros y validando rápidamente con un modelo de Naive Bayes, finalmente nos quedamos con n-gramas de 1 o 2 palabras, eliminando stop-words en inglés, con una frecuencia máxima de aparición del 20% y una mínima del 0,001%. Esto nos daba matrices con 34k columnas, lo que sólo se podía almacenar en matrices sparse, por lo que trabajamos con esta clase de matrices. Observamos también modelando rápidamente que no había prácticamente diferencias entre los resultados de BoW y Tf-idf, por lo que optamos por sólo usar el primero para reducir a la mitad el tiempo de modelado. 

### 2.2. NMF y LDA
Para reducir el tamaño de la matriz (y con eso el tiempo de procesamiento) o evitar la dimensión de la dimensionalidad probamos aplicar estos dos enfoques de reducción de dimensionalidad, pero no terminaron solucionando finalmente ninguno de estas cuestiones: la primera porque al reducir la dimensionalidad se pasaba de una matriz sparse a una densa, con lo cual paradójicamente terminaba aumentando el tamaño de la misma; mientras que en el segundo caso no se espeaba realmente tener maldicion de dimensionalidad ya que a pesar de la gran cantidad de features (34k), igualmente la gran cantidad de líneas (2M) hacía que la matriz tuviera una forma marcadamente "vertical", con lo que no se preveía este problema. Luego de un par de modelos que verificaron esto último, decidimos descartar estos enfoques.

### 2.3. Features de "Comportamiento"
Creamos en total 12 variables que llamamos de comportamiento como fueron: cantidad de palabras usadas, de preguntas, de sustantivos, de adjetivos, número de línea, etc. Para esto hicimos varias RegEx y usamos la librería TextBlob.

## 3. Modelado

### 3.0. Benchmark
Para tener un modelo de base con el cual comparamos corrimos un modelo simple con BoW + un clasificador de Naive Bayes, el cual corrió muy rápido y obtuvimos un valor del AUC de 0,6.

### 3.1. Primeros modelos
Sobre las matrices obtenidas en los puntos 2.1. y 2.3., luego de estandarizarlas aplicamos 6 modelos distintos con búsqueda de hiperparámetros mediante RandomSearch y utilizando CrossValidation, utilizando como criterio de selección el AUC. Los modelos aplicados fueron:

- Naive Bayes
- Logistic Regression
- Random Forest
- Ada Boost
- XGBoost
- LightGBM

Extrañamente, para la matriz de BoW obtuvimos un mejor resultado con los modelos más sencillos. En el caso de las features de comportamiento el resultado fue al revés, más parecido a lo esperado.

En este punto también quisimos probar hacer Undersampling u Oversampling dado el desbalanceo de las Labels. Lo segundo lo descartamos por el gran tiempo de procesamiento sumado al hecho que no es recomendado para datasets con tanto mayor número de muestras que de features; el segundo lo aplicamos con distintas distribuciones de labels, pero en ningún caso arrojó un mejor resultado que con el dataset original, por lo que también fue descartado.

### 3.2. Ensamble
Usando los outputs de los 12 modelos obtenidos en el punto 3.1. (2 matrices x 6 modelos), construímos un modelo que nos daba un mejor resultado (AUC) que cualquiera de los demás (0,925 en train vs 0,860 del siguiente mejor modelo). 

### 3.3. Selección de modelo y determinación del Umbral
De acuerdo a los valores del AUC de train optaríamos por el modelo de ensamble. Chequeando el AUC en el set de test (0,870) vemos que aunque haya un poco de overfitting (las pruebas que habíamos hecho antes del ensamble arrojaban menor diferencia entre train y test), sigue siendo claramente superior el modelo de ensamble a los demás. Para elegir el umbral fuimos optimizando el F3 de manera iterativa con una función que creamos hasta detenernos en un Umbral del 5% con el que obteníamos un valor de F3 = 0,549.

Teniendo el modelo y el valor del Umbral, los valores obtenidos fueron los siguientes:
- Recall           0,618
- Precision        0,163
- F1               0,257
- **F3               0,482**

A modo de comparación, cabe destacar que este valor es superior al obtenido por el equipo que ganó la competencia en aquel entonces, con un F3 de 0.476.






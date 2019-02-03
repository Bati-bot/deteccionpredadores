# Detección de Depredadores Sexuales en salas de chat
Trabajo integrador en el marco del curso Data Science de Digital House 
- por Juan Ferraro, Gabriel Casal y Santiago Paz

## Descripción General
*"Un depredador sexual es una persona que obtiene o trata de obtener contacto sexual con otra de manera abusiva. De forma análoga a cómo un depredador caza su presa, se cree que el depredador sexual "caza" a sus parejas sexuales."* (Wikipedia en inglés).

El propósito de este trabajo es la identificación de potenciales depredadores sexuales, según la definición arriba citada, a partir de características léxicas o de comportamiento discursivo. 

### Data
La base fue armada por PAN, una organización que fomenta la investigación forense de textos digitales mediante la organización de competencias. Este dataset formó parte de una competencia en 2012. El mismo cuenta con una base de 200.000 conversaciones que comprenden un total de 2.000.000 de registros (lineas de chat) únicos. Las conversaciones de los depredadores sexuales se obtuvieron del sitio Peverted Justice en el que voluntarios se hacen pasar por menores de edad para atrapar a los depredadores; mientras que las conversaciones regulares provienen de distintos sitios de chat. Para asemejarlo a la realidad, se balancearon las conversaciones de modo que sólo el 3% de las mismas correspondan a conversaciones con depredadores.

El evento constaba de dos competencias: la primera buscaba identificar conversaciones en las que participaran depredadores y la segunda  encontrar las líneas de los mismos. Nosotros optamos por esta última, por la mayor dificultad que representaba. Para este tipo de desafíos tan desbalanceados, medidas como el Accuracy no son efectivas, por lo que deben buscarse otras métricas como el area bajo la curva ROC (AUC). En este caso, se tomó como métrica el F3, medida que privilegia el Recall sobre la Presicion tratando de identificar el mayor número de depredadores aunque eso signifique clasificar así a algunos inocentes, por lo que será el F3 lo que trataremos de optimizar.

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

## El Trabajo


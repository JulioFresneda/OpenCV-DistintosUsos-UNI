# Visión por Computador
## Prácticas de cuarto curso de la asignatura de Visión por Computador | 2018-2019 


La visión por computador es el conjunto de herramientas y métodos que permiten obtener, procesar y analizar imágenes del mundo real con la finalidad de que puedan ser tratadas por un ordenador. Esto permite automatizar una amplia gama de tareas al aportar a las máquinas la información que necesitan para la toma de decisiones correctas en cada una de las tareas en las que han sido asignadas.

En este repositorio están contenidas diferentes ejercicios con imágenes, todos relacionados con esta ciencia.

### P1: Ejercicios relacionados con convoluciones, gaussianas, y pirámides laplacianas y gaussianas, así como imágenes híbridas
En esta primera práctica se realizan los siguientes ejercicios:
  -Cálculo de la convolución de imágenes con una máscara Gaussiana 2D, con distintos tamaños de máscara
  -Cálculo de las máscaras que permiten calcular convoluciones con máscaras de derivadas
  -Cálculo de convoluciones con una máscara de Laplaciana-de-Gaussiana de tamaño variable
  -Cálculo de pirámides gaussianas y laplacianas
  -Composición de imágenes híbridas, acoplando frecuencias altas de una imagen con frecuencias bajas de otra
  
Ejemplo: Imagen de Monroe convolucionada con tamaños de máscara y sigma variables
![Manroe Convolucionada](https://github.com/juliofgx17/VisionPorComputador/blob/master/monroe_convolucionada.png)


### P2: Detección de puntos SIFT y SURF.
En esta segunda práctica se realizan los siguientes ejercicios:
  -Aplicar la detección de puntos SIFT y SURF sobre las imágenes, representar dichos puntos sobre las imágenes
  -Usar el detector-descriptor SIFT de OpenCV sobre las imágenes (cv2.xfeatures2d.SIFT create()). Extraer sus listas de keyPoints y descriptores asociados. Establecer las correspondencias existentes entre ellos usando el objeto BFMatcher de OpenCV y los criterios de correspondencias BruteForce+crossCheck y Lowe-Average-2NN

### P3: Emparejamiento de descriptores, recuperación de imágenes y visualización del vocabulario
En esta tercera práctica se realizan los siguientes ejercicios:
  -Detectar y emparejar los descriptores de una región con los descriptores de otra imagen, para así ver si
podemos detectar partes iguales en escenas comunes de dos imágenes diferentes.
  -Implementar un modelo de índice invertido + bolsa de palabras para ciertas imágenes usando un vocabulario. Verificar que el modelo construido para cada imagen permite recuperar imágenes de la misma escena cuando la comparamos al resto de
imágenes de la base de datos.
  -Elegir palabras visuales de un diccionario, es decir, centroides de un clúster, y a partir de ahí obtener los x parches cuyos descriptores de una lista de descriptores sean los más cercanos a ese clúster.
  
Ejemplo: Emparejamiento de descriptores de una región de una imagen con descriptores de otra imagen
[!Descriptores Friends](https://github.com/juliofgx17/VisionPorComputador/blob/master/descriptores_friends.png)


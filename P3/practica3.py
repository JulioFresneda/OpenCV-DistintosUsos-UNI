#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 16:57:15 2018

@author: juliox
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import auxFunc as aux
from scipy.cluster.vq import vq
from scipy.spatial.distance import euclidean
import math



# Creamos el SIFT
sift = cv2.xfeatures2d.SIFT_create()


####################################################################################
#                     FUNCIONES GENERALES
####################################################################################



## Función para mostrar imágenes a escala real
####################################################################################
def imShowRealScale( img,gray=0, title='img', scale=1):
    width=img.shape[1]*scale # pixels
    height=img.shape[0]*scale
    margin=50 # pixels
    dpi=100. # dots per inch
    
    figsize=((width+2*margin)/dpi, (height+2*margin)/dpi) # inches
    left = margin/dpi/figsize[0] #axes ratio
    bottom = margin/dpi/figsize[1]
    
    fig = plt.figure(figsize=figsize, dpi=dpi)
    fig.subplots_adjust(left=left, bottom=bottom, right=1.-left, top=1.-bottom)
    plt.title(title)

    
    if( gray != 0 ):
        plt.imshow(img,cmap='gray')
    else:
        plt.imshow(img)
    plt.show()

###################################################################################
    
    
    
    
    
    
    
    
    
    

####################################################################################    
####################################################################################
#                   EJERCICIO 1: Emparejamiento de descriptores
####################################################################################    
####################################################################################    
 
"""
Este ejercicio consiste en emparejar los descriptores de una región con los
descriptores de otra imagen, para así ver si podemos detectar partes iguales
en escenas comunes.

Lo primero que hacemos es obtener los puntos del polígono que forma la región
que queremos, una vez tenemos los puntos creamos una máscara, y finalmente obtenemos la región.

Obtenemos los descriptores de los puntos SIFT de esa región, y los descriptores de la 
imagen donde queremos buscar esa región. Calculamos sus correspondencias, y las mostramos.

-Para obtener la región de la imagen, usamos getPoly()
-Para detectar y dibujar los matches entre la región y la imagen, usamos 
 matchesRegion(), la cual llama a detectMatchesKNN() 


"""    
    
    
    
    
####################################################################################
#                     FUNCIONES EJERCICIO 1
####################################################################################    

## Función que devuelve el polígono dados una imagen y unos puntos
def getPoly( img, points ):
    
    # Creamos una máscara vacía (negra)
    mask = np.zeros((img.shape[0], img.shape[1]))
    
    # Convertimos los puntos en un polígono con la función de OpenCV
    cv2.fillConvexPoly(mask, points, 1)
    
    # Pasamos la máscara a tipo booleano (se pinta píxel o no)
    mask = mask.astype(np.bool)
    
    # Finalmente con ayuda de la máscara, obtenemos sólo el polígono de la imagen
    out = np.zeros_like(img)
    out[mask] = img[mask]
    
    # Devolvemos el polígono
    return out


# Función para dibujar matches eligiendo según ratio test de Lowe.
# Utilizada en la práctica 2.
################         LOWE-AVERAGE-2NN             ##############################   
def detectMatchesKNN( img1, img2, kps,descs, num_matches = 10, ratio = 0.8, knn = 2 ):
    
    # Desempaquetamos los keypoints y descriptores asociados
    (kp1, kp2) = kps
    (desc1, desc2) = descs
    
    # Buscamos los knn mejores matches
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1,desc2, k= knn)
    
    # Aplicamos el test que Lowe propuso en su paper
    good_matches = []
    for m,n in matches:
        if m.distance < ratio*n.distance:
            good_matches.append([m])
    
    print(len(good_matches))
    
    # Devolvemos la imagen con los matches dibujados
    return (cv2.drawMatchesKnn(img1,kp1,img2,kp2,good_matches,None))
         



# Función para detectar correspondencias entre una región y una imagen distinta
def matchesRegion( img1, img2 ):
    
    # Usamos la función auxiliar dada en auxFunc.py para extraer una región con el ratón
    # La región la extraeremos de 'pareja1a', 'pareja2a', y 'pareja3a'
    pts = aux.extractRegion(img1)
    pts = np.asarray(pts)
    
    # Región seleccionada
    region = getPoly(img1,pts)
    
    # Detectamos los keypoints y descriptores de la región y la segunda imagen
    kp_region, desc_region = sift.detectAndCompute(region,None)
    kp_img2, desc_img2 = sift.detectAndCompute(img2,None)
    
    # Obtenemos los matches y los dibujamos en la imagen final 
    img_matches = detectMatchesKNN(img1,img2,(kp_region,kp_img2),(desc_region,desc_img2))
    
    return img_matches


    



####################################################################################
#                     EJECUCIÓN EJERCICIO 1
####################################################################################  
    
# Vamos a probar con las siguientes parejas de imágenes
    
# Dibujo de la camiseta de Rachel
pareja1a = cv2.imread('imagenes/210.png')
pareja1a = cv2.cvtColor(pareja1a,cv2.COLOR_BGR2RGB)

pareja1b = cv2.imread('imagenes/214.png')
pareja1b = cv2.cvtColor(pareja1b,cv2.COLOR_BGR2RGB)

plt.imshow(pareja1a)
plt.show()

plt.imshow(pareja1b)
plt.show()
input("Pulsa enter para continuar...")






# Llamamos a la función que detecte correspondencias
img_camiseta = matchesRegion(pareja1a,pareja1b)

# Como vemos, se encuentran correspondencias correctamente. Era fácil, ya que
# la camiseta tiene un dibujo muy característico
imShowRealScale(img_camiseta)
input("Pulsa enter para continuar...")




# Adorno de la pared
pareja2a = cv2.imread('imagenes/391.png')
pareja2a = cv2.cvtColor(pareja2a,cv2.COLOR_BGR2RGB)

pareja2b = cv2.imread('imagenes/385.png')
pareja2b = cv2.cvtColor(pareja2b,cv2.COLOR_BGR2RGB)


plt.imshow(pareja2a)
plt.show()

plt.imshow(pareja2b)
plt.show()
input("Pulsa enter para continuar...")

img_adorno = matchesRegion(pareja2a,pareja2b)
# Aquí vemos que, aunque en la segunda imagen no se ve la parte izquierda, 
# con la parte derecha es suficiente para que se hagan algunos matches correctos.
imShowRealScale(img_adorno)


input("Pulsa enter para continuar...")

# Pelota de baloncesto
pareja3a = cv2.imread('imagenes/81.png')
pareja3a = cv2.cvtColor(pareja3a,cv2.COLOR_BGR2RGB)

pareja3b = cv2.imread('imagenes/153.png')
pareja3b = cv2.cvtColor(pareja3b,cv2.COLOR_BGR2RGB)


plt.imshow(pareja3a)
plt.show()
plt.imshow(pareja3b)
plt.show()
input("Pulsa enter para continuar...")


img_pelota = matchesRegion(pareja3a,pareja3b)

# Aquí no se han encontrado correspondencias. Es normal, puesto que aunque 
# nosotros veamos claro la correspondencia, el algoritmo usa los gradientes,
# y los únicos gradientes que tiene la pelota es el borde, además de que 
# el fondo de la pelota es distinto en cada imágen
imShowRealScale(img_pelota)
input("Pulsa enter para continuar...")



# En conclusión, podemos decir que esta forma de buscar correspondencias es
# efectiva sólo cuando en la región hay patrones claros con gradientes precisos, pero
# no es efectiva cuando la región no tiene gradientes, como la pelota.



   
            



####################################################################################    
####################################################################################
#                                  EJERCICIO 2
####################################################################################    
####################################################################################  

"""
En este ejercicio vamos a crear una clase la cual dándole una imagen, nos
devuelve las imágenes más parecidas. Para ello usaremos las siguientes funciones:
    -generate_histogram() nos devuelve el histograma de una imagen
    -compare_histograms() compara dos histogramas
    -get_similar() obtiene los histogramas más parecidos a un histograma dado
    
Usaremos una clase, IndiceInvertido, ya que así sólo tendremos que generar un
histograma para cada imagen al inicializar el objeto. 

Para crear un objeto de esta clase se le piden como argumentos un 
diccionario y una batería de imágenes, y al inicializar, la clase crea
un histograma para cada imagen con generate_histogram().

Esta clase tiene un método GetSimilarImages(), que a partir de una imagen,
genera sus descriptores SIFT y su histograma, y con la función get_similar
obtenemos los índices de las imágenes cuyos histogramas son los más parecidos
al histograma de nuestra imagen.
    
    
Para este proceso vamos a usar el vocabulario dado en 'kmeanscenters2000.pkl'.
Este archivo contiene un diccionario con 2000 palabras o centroides, 
las cuales usaremos para generar los histogramas.

 -Función generate_histogram: 
  Para generar un histograma de una imagen, necesitamos un diccionario de
  palabras y los descriptores de la imagen. Si el diccionario tiene 2000 palabras,
  el histograma tendrá 2000 'barras' o columnas, las cuales representan a cada
  palabra del diccionario. Por cada descriptor de la imagen, encontramos a qué 
  palabra/centroide pertenece y aumentamos en 1 la 'barra' del histograma correspondiente 
  a ese centroide. Por ejemplo, si la imagen tiene 3 descriptores pertenecientes
  al centroide diccionario[7], histograma[7] = 3.
  
 -Función compare_histograms:
  Para comparar dos histogramas, usamos como medida de distancia el producto escalar,
  no hace falta normalizar ya que ya normalizamos a la hora de generar el histograma
  
 -Función get_similar:
  Para a partir de una imagen obtener las imágenes más parecidas, comparamos
  el histograma de esa imagen con todos los histogramas de las demás imágenes,
  usando compare_histograms, y seleccionamos los más cercanos


"""    




####################################################################################
#                     FUNCIONES EJERCICIO 2
####################################################################################  


  
# Función para generar el histograma dados los descriptores de una imagen,
# y un diccionario de palabras   
def generate_histogram( dictionary, desc ):
    
    # Creamos un histograma lleno de ceros
    histogram = np.zeros(len(dictionary))
    
    # Normalizamos tanto los descriptores como el diccionario
    norm_desc = desc
    norm_dict = dictionary
    cv2.normalize(src=desc,dst=norm_desc,norm_type=cv2.NORM_L2)
    cv2.normalize(src=dictionary,dst=norm_dict,norm_type=cv2.NORM_L2)
    
    
    # La siguiente función devuelve los índices de cada palabra del diccionario
    # que esté en los descriptores 'desc'. La he encontrado en el siguiente link:
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.vq.vq.html
    words, distances = vq(norm_desc, norm_dict)
    
    # Para cada uno de los índices de las palabras encontradas:
    for word in words:
        
        # Actualizamos el histograma sumándole 1 al índice correspondiente
        # a la palabra encontrada
        histogram[word] = histogram[word] + 1
        
        
    # Devolvemos el histograma 
    return histogram


# Función que compara dos histogramas. Aplica el producto escalar normalizado,
# sim(dj, q) = <dj,q>/(||dj||x||q||), el cual sirve como medida de distancia entre histogramas
def compare_histograms( a, b ):
    
    # Las longitudes deben ser iguales
    if( len(a) == len(b) ):
        
        # Inicializamos a 0
        dq = 0
        d = 0
        q = 0
        
        # Para cada "barra" del histograma:
        for i in range(len(a)):

            # Aplicamos la fórmula
            dq = dq + a[i]*b[i]
            d = d + a[i]*a[i]
            q = q +  b[i]*b[i]
        
        # Devolvemos el resultado final
        return dq / (math.sqrt(d)*math.sqrt(q))
    
    
    
    

# Dado un histograma, y un conjunto de histogramas, devuelve los n histogramas
# más similares al primer histograma
def get_similar( histogram, histograms, n ):
    
    if( n <= len(histograms)):
        
        # Distancias
        distances = []
        
        # En distancias almacenamos la distancia entre nuestro histograma y cada 
        # histograma del segundo conjunto, usando la función compare_histograms
        for i in range(len(histograms)):
            distances.append(compare_histograms(histogram,histograms[i]))
            
            
            
        # argsort devuelve los índices de un array si estuviera ordenado:
        # https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.argsort.html
        distances_index_sorted = np.argsort(np.array(distances))   
        
        # Como queremos los n mejores, y están ordenados de peor a mejor,
        # obtenemos los índices desde la última posición - n hasta la última posición
        res = distances_index_sorted[:len(distances_index_sorted)-n-1:-1]
        
        return res




# Una vez tenemos las funciones básicas, vamos a crear una clase de
# índice invertido, para que dado un índex de una imagen, obtengamos las 
# imágenes que más se parecen
    

class IndiceInvertido:
    
    # Inicializamos los objetos internos
    def __init__( self, diccionario, imagenes ):
        self.imagenes = imagenes
        self.diccionario = diccionario
        self.indice = []
        self.histograms = []


        # Obtenemos los histogramas de cada imagen, usando el diccionario         
        for img in self.imagenes:              
            k, d = sift.detectAndCompute(img,None)
            self.histograms.append( generate_histogram(self.diccionario, d) )
            
                
        # Cargamos en el índice cada una de las palabras
        for word in self.diccionario:
            self.indice.append([])
            
        # Completamos el índice
        # Para cada palabra del indice:
        for w in range(len(self.indice)):
            # Para cada barra del histograma
            for h in range(len(self.histograms)):
                # Si la barra no está a 0, es que esa imagen contiene esa palabra: Esa palabra está en esa imagen
                if( self.histograms[h][w] != 0 ): self.indice[w].append(h)
        
    # Método que nos devuelve las <num> imágenes más parecidas a la imagen
    # cuyo índice es pasado como argumento
    def GetSimilarImages( self, index, num ):
        
        # Obtenemos los descriptores de la imagen
        kp, desc = sift.detectAndCompute(self.imagenes[index],None)
        
        # Generamos su histograma
        histogram = generate_histogram(self.diccionario, desc)
        
        # Buscamos los <num> histogramas más similares, obteniendo los índices
        similar_index = get_similar(histogram, self.histograms, num)
        
        # Obtenemos las imágenes a partir de los índices anteriores
        similar_images = []
        for index in similar_index:
            similar_images.append(self.imagenes[index])
            
        # Devolvemos las imágenes    
        return similar_images
    
    def GetIndiceInvertido( self, index ):
        if( index > 0 and index < len(self.indice) ): return self.indice[index]
        
    
    
    
    
####################################################################################
#                     EJECUCIÓN EJERCICIO 2
####################################################################################      



# Lo primero es cargar las imágenes. Son 441 en total.
images = []
for i in range(0,441):

    ruta = 'imagenes/' + str(i) + '.png'
    img = cv2.imread(ruta)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    images.append(img)




# Cargamos el diccionario con la función auxiliar
accuracy, labels, dictionary = aux.loadDictionary('imagenes/kmeanscenters2000.pkl')


# Creamos un objeto IndiceInvertido
indiceInvertido = IndiceInvertido(dictionary,images)


# Vamos a probar a ver, por ejemplo, 3 imágenes que contienen la palabra 288
img288 = indiceInvertido.GetIndiceInvertido(288)

# No sabemos como es la palabra, pero vemos que funciona
for i in img288[20:23]:
    plt.title('Imagenes con la palabra nº 288')
    plt.imshow(images[i])
    plt.show()

input("Pulsa enter para continuar...")

# Vamos a ver 5 imágenes parecidas a la imagen 91
# Pasamos 6 como argumento porque también obtenemos la imagen original
similars91 = indiceInvertido.GetSimilarImages(91,6) 

# Mostramos las imágenes en pantalla.
# La primera imagen que obtenemos es la misma que la original, por lo que no la mostramos.
# Las siguientes tres imágenes son bastante parecidas, evidentemente de la misma
# escena. Las dos siguientes son un poco distintas, pero sigue saliendo
# Joey con su camisa hawaiana, por lo que podemos ver que en este ejemplo
# el modelo es muy efectivo, aunque hay que decir que las imágenes son muy parecidas

plt.title('Original')
plt.imshow(similars91[0])
plt.show()

for s in similars91[1:]:
    plt.title('Similar')
    plt.imshow(s)
    plt.show()
    
input("Pulsa enter para continuar...")
    



# Vamos a ver 5 imágenes parecidas a la imagen 8
similars8 = indiceInvertido.GetSimilarImages(8,6) 

# Mostramos las imágenes en pantalla.
# En este caso, vemos que las 5 imágenes tienen un factor común con la original:
# En todas sale el sofá. Vemos que da igual quien esté sentado, reconoce
# del sofá y la habitación en general perfectamente, incluso en escenas de
# temporadas distintas. En este caso vemos que el modelo es bastante efectivo.
plt.title('Original')
plt.imshow(similars8[0])
plt.show()

for s in similars8[1:]:
    plt.title('Similar')
    plt.imshow(s)
    plt.show()
    
input("Pulsa enter para continuar...")   



# Vamos a ver 5 imágenes parecidas a la imagen 324
similars324 = indiceInvertido.GetSimilarImages(324,6) 

# Mostramos las imágenes en pantalla.
# En este último caso, nuestro último modelo no ha sido nada efectivo.
# Ninguna de las supuestas imágenes más parecidas se parecen en nada a la original.
# En la última salen también Ross y Monica, pero parece casualidad.
# Esto es porque aunque en la original nosotros vemos claramente a Ross y Monica,
# es muy difícil sacar keypoints de esa imagen, no hay nada especialmente relevante,
# (en el primer ejemplo tenemos la camisa hawaiana con claros patrones de Joey, 
# y en el segundo tenemos el color característico y los bordes del sofá).
plt.title('Original')
plt.imshow(similars324[0])
plt.show()

for s in similars324[1:]:
    plt.title('Similar')
    plt.imshow(s)
    plt.show()
    
input("Pulsa enter para continuar...")
    
# En conclusión, podemos decir que este método obtiene muy buenos resultados 
# cuando hay patrones claros en la imagen original, pero no tan buenos cuando
# la imagen es más genérica y sin nada especial que destaque.




####################################################################################    
####################################################################################
#                                  EJERCICIO 3
####################################################################################    
####################################################################################  
    
"""
En este ejercicio se nos pide elegir palabras visuales de un diccionario, es decir,
centroides de un clúster, y a partir de ahí obtener los x parches cuyos
descriptores de una lista de descriptores sean los más cercanos a ese clúster.

Se ha decidido hacer una clase para esta tarea, de forma que la mayoría de operaciones
solo se realizan al inicializar la clase.

Para realizar este ejercicio, hay que entender bien qué contiene cada archivo.
 -'kmeanscenter2000.pkl' contiene la distancia (no nos interesa), las etiquetas
  o labels y el diccionario de 2000 palabras.
  
  Cada palabra del diccionario es un centroide de un clúster, por lo cual cada
  posición del diccionario corresponde a un clúster distinto. Por ejemplo, 
  con diccionario[1980] obtenemos el centroide del clúster 1980.
  
  Labels es una lista con 193041 etiquetas. Está relacionado con el otro archivo.
  
  -'descriptorsAndpatches2000.pkl' contiene una lista con 193041 descriptores y 
  sus 193041 parches asociados. Estos descriptores y parches se han obtenido de
  las 441 imágenes de que tenemos.
  
  Para obtener a qué clúster pertenece cada descriptor, usamos la lista labels.
  
  Cada posición de labels corresponde a la posición del descriptor, y el contenido
  de esa posición de labels corresponde al clúster de ese descriptor. Por ejemplo,
  si queremos saber a qué cluster pertenece descriptores[4], accedemos a
  labels[4]. Si por ejemplo labels[4] = 1999, descriptores[4] pertenece al
  clúster 1999, cuyo centroide está en diccionario[1999].
  
Una vez descrito qué contiene cada archivo, pasamos a resolver el ejercicio.

Lo primero que hace nuestra clase es inicializar una lista clusters.
Esta lista separa los descriptores por clústers. Pero se guarda el descriptor en sí,
se guarda su índice en la lista y su distancia. Por ejemplo, si los descriptores 
5, 16000 y 99000 pertenecen al clúster 1220, clusters[1220] contendría los pares
(5,distancia), (16000,distancia) y (99000,distancia).

Una vez hecho ésto, para cada clúster calculamos la mediana de distancia de los 
descriptores de un clúster respecto a su centroide. Esto no es necesario para 
el ejercicio, pero nos vendrá bien para elegir palabras donde se vea que 
los parches son parecidos (mediana baja) o no tienen mucho que ver (mediana alta).

Para obtener los índices de los parches más cercanos a un clúster en concreto,
usamos getSimilarPatches(). Este método ordena por distancia los elementos de 
nuestro clúster de la lista de clústeres, y devuelve los índices de los <n>
con menor distancia.

Para obtener el índice del clúster con mejor mediana dado un rango, usamos 
getClusterByMedian(). Este método ordena los clústeres por su mediana, y devuelve
los índices de los <n> clústers con menor mediana que cumplan los mínimos y máximos
de mediana dados.


"""

####################################################################################
#                     FUNCIONES EJERCICIO 3
####################################################################################  



class ParchesCercanos:
    
    # Inicializamos la clase
    def __init__(self, desc, patches, labels, dictionary ):
        self.desc = desc # 193041 descriptores
        self.patches = patches # 193041 parches
        self.labels = labels # 193041 labels
        self.dictionary = dictionary # 2000 palabras
        
        # Vamos a crear una lista de clusters, uno por cada centroide del
        # diccionario. En cada posición habrá un par (index del desc, distancia con centroide)
        self.clusters = []

        # Inicializamos la lista
        for i in range(max(labels)[0]+1):
            self.clusters.append([])


        # Por cada uno de los descriptores
        for i in range(len(desc)):
    
            # Obtenemos su número de cluster
            label = labels[i][0]
            
            # Obtenemos la distancia euclídea entre el descriptor y el centroide de su clúster
            #https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.spatial.distance.euclidean.html
            distance = euclidean(dictionary[label], desc[i])
            
            # Añadimos a la lista de clusters su index en desc y su distancia con el centroide
            # Lo añadimos en la posición correspondiente al número de centroide al que pertenece
            self.clusters[label].append((i,distance))
            
            
        # Vamos a guardar una lista de medianas de distancia de los descriptores
        # de un clúster con su centroide, para ver qué descriptores
        # se asemejan de mediana más a sus centroides 
        self.medians = []    
    
        # Para cada clúster
        for c in self.clusters:
            
            # Obtenemos la distancia de cada descriptor con el centroide
            med = []
            for d in c:
                med.append(d[1])
            
            # Hacemos la mediana y la almacenamos
            self.medians.append(np.median(med))
            
            
            
        
    # Método para obtener los index de los <num> parches más similares a un centroide dado
    def getSimilarPatches( self, index, num ):
        
        # Aquí se almacenan los índices 
        patches_index = []
        
        # Obtenemos el cluster objetivo
        cluster = self.clusters[index]
        
        # Vamos a obtener las distancias
        distancias = []
                
        # Para cada descriptor del cluster, obtenemos su distancia y su indice
        for desc in cluster:
            distancias.append(desc[1])
            patches_index.append(desc[0])
        
        # Obtenemos los índices ordenados de la lista de distancias
        sort_index = np.argsort(distancias)
        
        # Aquí se almacena el resultado
        best_patches = []
        
        # Si hay más parches que el número pedido
        if( num <= len(sort_index) ):
            
            # <num> veces:
            for i in range(num):
                # Obtenemos el índice del parche en la posición dada en sort_index
                best_patches.append(patches_index[sort_index[i]])
                
        else:
            for i in len(sort_index):
                best_patches.append(patches_index[sort_index[i]])
                
                
        # Devolvemos los índices
        return best_patches
             
    
    
    
    # Este método devuelve los <num> índices de cluster con mejor mediana entre los 
    # intervalos pasados como atributos.
    # Este método nos va a servir para que en el ejercicio usemos clústers
    # donde sus descriptores se asemejen bastante, y se vea visualmente el
    # parecido
    def getClusterByMedian( self, num, max_median = 1, min_median = 0 ):
        
        # Obtenemos los índices ordenados de las medianas
        median_index = np.argsort(self.medians)
        
        # Lista con los índices que pasan los umbrales dados
        index_approved = []
        
        # Vamos comprobando las medias de mejor a peor
        for i in range(len(median_index)):
            # Si la mediana cumple los umbrales, guardamos su índice
            if( self.medians[median_index[i]] <= max_median and self.medians[median_index[i]] >= min_median ):
                index_approved.append(median_index[i])
                
                
        if( num < len(index_approved) ):
            return index_approved[:num]
        else: return index_approved
        
                
        
                    
####################################################################################
#                     EJECUCIÓN EJERCICIO 3
####################################################################################              
                        
                



# Abrimos los archivos
desc, patches = aux.loadAux('imagenes/descriptorsAndpatches2000.pkl', True)
accuracy, labels, dictionary = aux.loadDictionary('imagenes/kmeanscenters2000.pkl')

# Creamos el objeto ParchesCercanos
parchesCercanos = ParchesCercanos(desc,patches,labels,dictionary)


# Vamos a elegir tres palabras visuales diferentes


# Primero vamos a buscar un clúster cuya distancia con los descriptores sea
# mínima

cl = parchesCercanos.getClusterByMedian(1)

# Nos ha devuelto el cluster 850. Vamos a calcular sus 10 parches más cercanos
patches_index = parchesCercanos.getSimilarPatches(cl[0],10)

for i in patches_index:
    imShowRealScale(patches[i],title=str(i))
    
input("Pulsa enter para continuar...")

# Como podemos ver, los 10 parches son casi idénticos, lo cual era de 
# esperar pues hemos cogido al mejor clúster posible. 


# Vamos a probar ahora con un clúster algo menor: Alguno cuya mediana de
# distancias de sus descriptores sea mayor que 0.2, por ejemplo.
cl = parchesCercanos.getClusterByMedian(1,min_median=0.2)

  
# Nos ha devuelto el cluster 1933. Vamos a calcular sus 10 parches más cercanos
patches_index = parchesCercanos.getSimilarPatches(cl[0],10)

for i in patches_index:
    imShowRealScale(patches[i],title=str(i))
    
input("Pulsa enter para continuar...")
    
# Este clúster es maś interesante, pues aunque vemos que no todos los 
# parches son iguales, todos comparten unos patrones:
# Fondo negro y una forma de 'T' de color gris.
    
    
    
# Vamos ahora a obtener algún clúster algo malo, alguno cuya media
# sea mayor de 0.5
cl = parchesCercanos.getClusterByMedian(1,min_median=0.5)

  
# Nos ha devuelto el cluster 1974. Vamos a calcular sus 10 parches más cercanos
patches_index = parchesCercanos.getSimilarPatches(cl[0],10)

for i in patches_index:
    imShowRealScale(patches[i],title=str(i))
    
input("Pulsa enter para continuar...")
    
# Vemos que en los parches de este clúster apenas vemos parecidos.
# Como mucho podemos ver que las esquinas y la parte superior
# e inferior son algo más oscuras, pero aun así no se parecen demasiado, lo cual era esperable.
    
    
    





























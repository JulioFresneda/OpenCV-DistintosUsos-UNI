#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Julio Fresneda García
"""

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

## Función para mostrar imágenes a escala real
####################################################################################
def imShowRealScale(img,gray=0,title='img',scale=1):
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











"""
EJERCICIO 1
Detección de puntos SIFT y SURF. Aplicar la detección de
puntos SIFT y SURF sobre las imágenes, representar dichos puntos sobre
las imágenes haciendo uso de la función drawKeyPoints. Presentar los
resultados con las imágenes Yosemite.rar.
"""


#####################################################################################
#####################################################################################
                            # FUNCIONES EJERCICO 1 #
#####################################################################################
#####################################################################################





"""
Función para hacer unpack de cada octava. Fuente:
https://stackoverflow.com/questions/48385672/opencv-python-unpack-sift-octave
"""    

### El atributo octava de los keypoints obtenidos usando SIFT están
### enpaquetados en octava, capa y escala. Esta función lo desempaqueta
################################################################
def unpackSIFTOctave(kpt):

    _octave = kpt.octave
    octave = _octave&0xFF
    layer = (_octave>>8)&0xFF
    if octave>=128:
        octave |= -128
    if octave>=0:
        scale = float(1/(1<<octave))
    else:
        scale = float(1<<-octave)
    return (octave, layer, scale)
################################################################



# Método para dibujar KeyPoints mediante SIFT
####################################################################################
####################################################################################
def detectKeyPointsSIFT( img, sift ):

    # Detectamos los puntos sift de la imagen
    kp_sift, desc_sift = sift.detectAndCompute(img,None)
    
    # Dibujamos estos puntos en una copia de la imagen
    imgkp = img.copy()
    imgkp = cv.drawKeypoints(imgkp,kp_sift,imgkp)
    
    
    
    
    # Vamos a desempaquetar las octavas
    unpacked_kp_sift = []
    for i in range(0,len(kp_sift)):
        unpacked_kp_sift.append(unpackSIFTOctave(kp_sift[i]))
        
    
    ## Num de octavas y capas. Vamos a obtener el máximo número de octavas
    ## y de paso el mínimo (es -1, pero lo implemento para asegurar)
    min_oct = unpacked_kp_sift[0][0]
    max_oct = unpacked_kp_sift[0][0]
    
    min_lay = unpacked_kp_sift[0][1]
    max_lay = unpacked_kp_sift[0][1]
    
    for i in range(len(unpacked_kp_sift)):
        if( unpacked_kp_sift[i][0] > max_oct ):
            max_oct = unpacked_kp_sift[i][0]
        if( unpacked_kp_sift[i][0] < min_oct ):
            min_oct = unpacked_kp_sift[i][0]
            
        if( unpacked_kp_sift[i][1] > max_lay ):
            max_lay = unpacked_kp_sift[i][0]
        if( unpacked_kp_sift[i][0] < min_lay ):
            min_lay = unpacked_kp_sift[i][0]
            
            
    num_oct = max_oct - min_oct +1 # Les sumamos el 0
    num_lay = max_lay - min_lay +1
    
    
    # Listas para distintas octavas
    lista_octavas = []
    for i in range(num_oct):
        temp = []
        lista_octavas.append(temp)
    
    # Añadimos a la lista de octavas cada octava. para acceder por ejemplo a
    # la octava -1, usamos la posición [0]. Para acceder a la 2, la posición [3], etc.
    for i in range(len(unpacked_kp_sift)):
        #print(str(unpacked_kp[i][0]) + " " + str(min_oct) + " " + str(unpacked_kp[i][0]-min_oct))
        lista_octavas[unpacked_kp_sift[i][0]-min_oct].append(kp_sift[i])
        
        
        
    """
    (b) Identificar cuántos puntos se han detectado dentro de cada octava.
    En el caso de SIFT, identificar también los puntos detectados en
    cada capa. Mostrar el resultado dibujando sobre la imagen original
    un cı́rculo centrado en cada punto y de radio proporcional al valor de
    sigma usado para su detección (ver circle()) y pintar cada octava
    en un color.
    """
    # He usado la misma sigma para cada octava, por lo tanto no 
    # cambiaré el radio.   
    
    # Vamos a ver cuantos puntos tiene cada octava
    print("SIFT: Numero de Keypoints: " + str(len(kp_sift)))
    print("Numero de puntos para cada octava:")
    for i in range(len(lista_octavas)):
        print("Octava " + str(i) + ": " + str(len(lista_octavas[i])))
        
        
    ## Pintamos keypoints donde cada octava tiene un color
    img_oct = img.copy()
    for i in range(len(lista_octavas)):
        
        # Convertimos los keypoints en puntos 2d
        points2d = cv.KeyPoint_convert(lista_octavas[i])  
        
        if( i == 0 ): color = (255,0,0)
        if( i == 1 ): color = (0,255,0)
        if( i == 2 ): color = (0,0,255)
        if( i == 3 ): color = (255,255,0)
        if( i == 4 ): color = (255,0,255)
        if( i == 5 ): color = (0,255,255)
        if( i == 6 ): color = (0,0,0)
        if( i > 6 ): color = (50*i,0,abs(255-20*i))
        
        # Recorremos todos los puntos de la octava actual y los pintamos
        for j in range(len(points2d)):
            cv.circle(img_oct,(points2d[j][0],points2d[j][1]), 5, color, 1)
      
    
    
   
    
    
    
    
    # Listas para distintas capas. Hacemos lo mismo que hicimos con las octavas
    lista_capas = []
    for i in range(num_lay):
        temp = []
        lista_capas.append(temp)
    
    
    for i in range(len(unpacked_kp_sift)):
        #print(str(unpacked_kp[i][0]) + " " + str(min_oct) + " " + str(unpacked_kp[i][0]-min_oct))
        lista_capas[unpacked_kp_sift[i][1]-min_lay].append(kp_sift[i])
        
        
        
    # Vamos a ver cuantos puntos tiene cada capa
    print("Numero de puntos para cada capa:")
    for i in range(len(lista_capas)):
        print("Capa " + str(i) + ": " + str(len(lista_capas[i])))    
        
        
        
        
    ## Pintamos keypoints por cada capas 
    img_lay = img.copy()
        
    for i in range(len(lista_capas)):
        points2d = cv.KeyPoint_convert(lista_capas[i])  
        
        if( i == 0 ): color = (255,0,0)
        if( i == 1 ): color = (0,255,0)
        if( i == 2 ): color = (0,0,255)
        if( i == 3 ): color = (255,255,0)
        if( i == 4 ): color = (255,0,255)
        if( i == 5 ): color = (0,255,255)
        if( i == 6 ): color = (0,0,0)
        if( i > 6 ): color = (50*i,0,abs(255-20*i))
        
        for j in range(len(points2d)):
            cv.circle(img_lay,(points2d[j][0],points2d[j][1]), 5, color, 1)
      
    
    # Devolvemos las imágenes con los keypoints pintados, con los keypoints coloreados
    # según octavas y según capas, y los keypoints y descriptores
    return (imgkp, img_oct, img_lay, kp_sift, desc_sift )
####################################################################################
####################################################################################    
    


# Método para dibujar KeyPoints mediante SURF
####################################################################################
####################################################################################  
def detectKeyPointsSURF( img, surf ):

    # Detectamos los puntos SURF de la imagen
    kp_surf = surf.detect(img,None)
    
    # Dibujamos estos puntos
    imgkp = img.copy()
    imgkp = cv.drawKeypoints(imgkp,kp_surf,imgkp)
    

    
    ## Num de octavas. En SURF las octavas no vienen empaquetadas.
    min_oct = kp_surf[0].octave
    max_oct = kp_surf[0].octave
    
    # Buscamos la octava más pequeña y la más grande
    for i in range(len(kp_surf)):
        if( kp_surf[i].octave > max_oct ):
            max_oct = kp_surf[i].octave
        if( kp_surf[i].octave < min_oct ):
            min_oct = kp_surf[i].octave

            
            
    num_oct = max_oct - min_oct +1

    
    
    # Listas para distintas octavas
    lista_octavas = []
    for i in range(num_oct):
        temp = []
        lista_octavas.append(temp)
    
    
    for i in range(len(kp_surf)):
        lista_octavas[kp_surf[i].octave-min_oct].append(kp_surf[i])
        
        
    """
    (b) Identificar cuántos puntos se han detectado dentro de cada octava.
    En el caso de SIFT, identificar también los puntos detectados en
    cada capa. Mostrar el resultado dibujando sobre la imagen original
    un cı́rculo centrado en cada punto y de radio proporcional al valor de
    sigma usado para su detección (ver circle()) y pintar cada octava
    en un color.
    """
    # He usado la misma sigma para cada octava, por lo tanto no 
    # cambiaré el radio. 
    # Vamos a ver cuantos puntos tiene cada octava
    print("SURF: Numero de Keypoints: " + str(len(kp_surf)))
    print("Numero de puntos para cada octava:")
    for i in range(len(lista_octavas)):
        print("Octava " + str(i) + ": " + str(len(lista_octavas[i])))    
        
        
    ## Pintamos keypoints por cada octava 
    img_oct = img.copy()
    
    for i in range(len(lista_octavas)):
        points2d = cv.KeyPoint_convert(lista_octavas[i])  
        
        if( i == 0 ): color = (255,0,0)
        if( i == 1 ): color = (0,255,0)
        if( i == 2 ): color = (0,0,255)
        if( i == 3 ): color = (255,255,0)
        if( i == 4 ): color = (255,0,255)
        if( i == 5 ): color = (0,255,255)
        if( i == 6 ): color = (0,0,0)
        if( i > 6 ): color = (50*i,0,abs(255-20*i))
        
        for j in range(len(points2d)):
            cv.circle(img_oct,(points2d[j][0],points2d[j][1]), 5, color, 1)
      
    
    # Devolvemos la imágen con los keypoints, la imagen con los keypoints según octava, y el número de keypoints
    return (imgkp, img_oct, len(kp_surf))
    
####################################################################################
####################################################################################  






"""
(a) Variar los valores de umbral de la función de detección de puntos
hasta obtener un conjunto numeroso (≥ 1000) de puntos SIFT y
SURF que sea representativo de la imagen. Justificar la elección
de los parámetros en relación a la representatividad de los puntos
obtenidos.
"""


# Probamos con Yosemite1
img1 = cv.imread('imagenes/Yosemite1.jpg')

#### SIFT

# contrastThreshold es el umbral de contraste utilizado para filtrar las 
# características débiles en regiones de bajo contraste. 
# Cuanto mayor sea el umbral, menos características son producidas por el 
# detector.

# edgeThreshold es el umbral donde aceptamos o no las características en bordes.
# A diferencia del umbral anterior, aquí cuanto mayor sea el umbral, más características aceptaremos

# Vamos a ajustar para obtener alrededor de 1000 keypoints.
# Con los valores por defecto, obtenemos más de 2000, por lo que puede ser que algunos no sean
# demasiado relevantes. Creo que los bordes pueden ser más representativos que las zonas de
# bajo contraste, por lo tanto seremos más restrictivos con el umbral de contraste.
sift = cv.xfeatures2d.SIFT_create(contrastThreshold=0.08,edgeThreshold=10)

# Obtenemos nuestras imágenes con los keypoints dibujados para la primera imagen
(imgkp, img_oct, img_lay, kp_sift_img1, desc_sift_img1) = detectKeyPointsSIFT( img1, sift )



imShowRealScale(imgkp,title='Yosemite1, keypoints SIFT')
imShowRealScale(img_oct,title='Yosemite1, keypoints SIFT por octavas')
imShowRealScale(img_lay,title='Yosemite1, keypoints SIFT por capas')

# 1238 keypoints



# Segunda imagen
img2 = cv.imread('imagenes/Yosemite2.jpg')
(imgkp, img_oct, img_lay, kp_sift_img2, desc_sift_img2) = detectKeyPointsSIFT( img2, sift )

imShowRealScale(imgkp,title='Yosemite2, keypoints SIFT')
imShowRealScale(img_oct,title='Yosemite2, keypoints SIFT por octavas')
imShowRealScale(img_lay,title='Yosemite2, keypoints SIFT por capas')

# 1146 keypoints



# Visualmente parece que los keypoints representan a las imágenes




#### SURF


# Igual que con SIFT, buscamos obtener algo más de 1000 keypoints. Para ello
# vamos a ajustar el Hessian Threshold. Sólo las características cuyo hessian
# sea mayor que el límite puesto se conservarán. A más valor, menos características.
surf = cv.xfeatures2d.SURF_create(700)

(imgkp, img_oct, num_kp) = detectKeyPointsSURF( img1, surf )

imShowRealScale(imgkp,title='Yosemite1, keypoints SURF')
imShowRealScale(img_oct,title='Yosemite1, keypoints SURF por octavas')

# 1246 keypoints



(imgkp, img_oct, num_kp) = detectKeyPointsSURF( img2, surf )

imShowRealScale(imgkp,title='Yosemite2, keypoints SURF')
imShowRealScale(img_oct,title='Yosemite2, keypoints SURF por octavas')

# 1034 keypoints







"""
Obtenemos descriptores
"""

"""
(c) Mostrar cómo con el vector de keyPoint extraı́dos se pueden calcu-
lar los descriptores SIFT y SURF asociados a cada punto usando
OpenCV.
"""

# No he conseguido obtener los descriptores a través de los Keypoints,
# pero se pueden obtener ambos a la vez con el siguiente método

(kp_sift1, desc_sift1) = sift.detectAndCompute(img1,None)
(kp_surf1, desc_surf1) = surf.detectAndCompute(img1,None)

(kp_sift2, desc_sift2) = sift.detectAndCompute(img2,None)
(kp_surf2, desc_surf2) = surf.detectAndCompute(img2,None)







"""
EJERCICIO 2
Usar el detector-descriptor SIFT de OpenCV sobre las imágenes
de Yosemite.rar (cv2.xfeatures2d.SIFT create()). Extraer sus lis-
tas de keyPoints y descriptores asociados. Establecer las corresponden-
cias existentes entre ellos usando el objeto BFMatcher de OpenCV y los
criterios de correspondencias “BruteForce+crossCheck y “Lowe-Average-
2NN”. (NOTA: Si se usan los resultados propios del puntos anterior en
lugar del cálculo de SIFT de OpenCV se añaden 0.5 puntos)




"""

#####################################################################################
#####################################################################################
                            # FUNCIONES EJERCICO 2 #
#####################################################################################
#####################################################################################


# Método para detectar matches usando fuerza bruta con crosscheck
################      BRUTAL FORCE WITH CROSSCHECK    ##############################   
####################################################################################  
def detectMatchesBF( img1, img2, kps, descs, num_matches ):
    
    # Desempaquetamos los keypoints y descriptores asociados
    (kp1, kp2) = kps
    (desc1, desc2) = descs
    
    # Buscamos los matches. Crosscheck activado
    bf = cv.BFMatcher(crossCheck = True)
    matches = bf.match(desc1,desc2)
    
    # Nos piden que mostremos 100 aleatorios
    random_matches = []
    from random import randint
    usados = []
    for i in range(num_matches):
        r = randint(0,len(matches)-1)
        while r in usados:
            r = randint(0,len(matches)-1)
        
        usados.append(r)
        random_matches.append(matches[r])
    
    
    # Dibujamos los matches
    imgfinal = cv.drawMatches(img1,kp1,img2,kp2,random_matches,None)
    
    return (imgfinal, matches)
####################################################################################
#################################################################################### 
    

# Método para buscar matches eligiendo según ratio test de Lowe.
################         LOWE-AVERAGE-2NN             ##############################   
####################################################################################  
def detectMatchesKNN( img1, img2, kps,descs, num_matches = 100, ratio = 0.5, knn = 2 ):
    
    # Desempaquetamos los keypoints y descriptores asociados
    (kp1, kp2) = kps
    (desc1, desc2) = descs
    
    # Buscamos los knn mejores matches
    bf = cv.BFMatcher()
    matches = bf.knnMatch(desc1,desc2, k= knn)
    
    # Aplicamos el test que Lowe propuso en su paper
    good_matches = []
    for m,n in matches:
        if m.distance < ratio*n.distance:
            good_matches.append([m])
    
    
    # Nos piden solo 100 aleatorios
    random_matches = []
    from random import randint
    usados = []
    for i in range(num_matches):
        r = randint(0,len(good_matches)-1)
        while r in usados:
            r = randint(0,len(good_matches)-1)
            
        usados.append(r)        
        random_matches.append(good_matches[r])
    
    
    return (cv.drawMatchesKnn(img1,kp1,img2,kp2,random_matches,None), good_matches)
####################################################################################
#################################################################################### 



"""
(a) Mostrar ambas imágenes en un mismo canvas y pintar lı́neas de difer-
entes colores entre las coordenadas de los puntos en correspondencias.
Mostrar en cada caso 100 elegidas aleatoriamente.
"""

# Vamos a usar los keypoints y descriptores usados en el apartado anterior
img_matches_bf, matches_bf = detectMatchesBF(img1,img2,(kp_sift_img1,kp_sift_img2),(desc_sift_img1,desc_sift_img2),100)
imShowRealScale(img_matches_bf,title='Matches con Brutal Force')

img_matches_knn, matches_knn = detectMatchesKNN(img1,img2,(kp_sift_img1,kp_sift_img2),(desc_sift_img1,desc_sift_img2),100)
imShowRealScale(img_matches_knn,title='Matches con Lowe ratio')


"""
(b) Valorar la calidad de los resultados obtenidos en términos de las corre-
spondencias válidas observadas por inspección ocular y las tendencias
de las lı́neas dibujadas.
"""

# Al tener las dos imágenes la misma altura vemos si la correspondencia
# es correcta si las líneas son horizontales. El resultado con el primer
# método es bueno, hay muchas líneas horizontales, sin embargo se equivoca
# en unas cuantas. En cambio con el segundo método el resultado parece perfecto,
# todas las líneas son horizontales. 


"""
(c) Comparar ambas técnicas de correspondencias en términos de la cal-
idad de sus correspondencias (suponer 100 aleatorias e inspección
visual).
"""

# Vistos los resultados, podemos decir que el segundo método es 
# notablemente mejor, cosa esperable, pues en este segundo método
# usamos el ratio test de Lowe, donde nos quedamos con los mejores
# matches.

# Un detalle que nos muestra que el segundo método es mejor es que el primer
# método hace matches con keypoints del borde izquierdo, keypoints que es imposible
# que estén en la imagen derecha pues en la imagen derecha el borde izquierdo de la otra
# imagen no aparece





"""
3. (2.5 puntos) Escribir una función que genere un mosaico de calidad a
partir de N = 3 imágenes relacionadas por homografı́as, sus listas de
keyPoints calculados de acuerdo al punto anterior y las correspondencias
encontradas entre dichas listas. Estimar las homografı́as entre ellas usando
la función cv2.findHomography(p1,p2, CV RANSAC,1). Para el mosaico
será necesario.

"""

#####################################################################################
#####################################################################################
                            # FUNCIONES EJERCICO 3 #
#####################################################################################
#####################################################################################



# Método que genera un mosaico entre tres imágenes
################                 MOSAIC3               ##############################   
####################################################################################  

def mosaic3( img1, img2, img3, ratio=0.75, reprojThresh=4.0 ):
    

    # Obtenemos keypoints y descriptores
    sift = cv.xfeatures2d.SIFT_create()




    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    kp3, des3 = sift.detectAndCompute(img3,None)
    
    
    # Empezamos por la derecha
    
    # Obtenemos matches de las dos imágenes de la derecha
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des3,des2, k=2)
    
    
    # Aplicamos ratio test
    good = []
    for m in matches:
        if m[0].distance < ratio*m[1].distance:
                good.append(m)
    matches = np.asarray(good)
    
    
    # Pasamos los keypoints a puntos en el espacio
    src_d = np.float32([ kp3[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
    dst_d = np.float32([ kp2[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)


    # Obtenemos la homografía de las dos imágenes de la derecha
    (H, status) = cv.findHomography(src_d, dst_d, cv.RANSAC, 1.0)
    
    
    # Dibujamos estas imágenes
    dst_d = cv.warpPerspective(img3,H,(img2.shape[1] + img3.shape[1], img2.shape[0]),borderMode=cv.BORDER_TRANSPARENT)
    dst_d[0:img2.shape[0], 0:img2.shape[1]] = img2
    
    

    # Terminamos con la imagen de la izquierda
    kp_d, des_d = sift.detectAndCompute(dst_d,None)

    # Obtenemos matches de las dos imágenes de la derecha
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des_d,des1, k=2)
    
    
    # Aplicamos ratio test
    good = []
    for m in matches:
        if m[0].distance < ratio*m[1].distance:
                good.append(m)
    matches = np.asarray(good)
    
    # Pasamos los keypoints a puntos en el espacio
    src = np.float32([ kp_d[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
    dst = np.float32([ kp1[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)


    # Obtenemos la homografía de la primera imagen y las otras dos
    (H, status) = cv.findHomography(src, dst, cv.RANSAC, 1.0)
    
    
    # Dibujamos estas imágenes
    dst = cv.warpPerspective(dst_d,H,(img1.shape[1] + dst_d.shape[1], img1.shape[0]),borderMode=cv.BORDER_TRANSPARENT)
    dst[0:img1.shape[0], 0:img1.shape[1]] = img1

    return dst





# Vamos a cargar las imágenes
img1 = cv.imread('imagenes/mosaico002.jpg')
img2 = cv.imread('imagenes/mosaico003.jpg')
img3 = cv.imread('imagenes/mosaico004.jpg')
img4 = cv.imread('imagenes/mosaico005.jpg')
img5 = cv.imread('imagenes/mosaico006.jpg')
img6 = cv.imread('imagenes/mosaico007.jpg')
img7 = cv.imread('imagenes/mosaico008.jpg')
img8 = cv.imread('imagenes/mosaico009.jpg')
img9 = cv.imread('imagenes/mosaico010.jpg')
img10 = cv.imread('imagenes/mosaico011.jpg')



# Vamos a hacer el mosaico con tres imágenes
res = mosaic3(img1,img2,img3)
imShowRealScale(res,title='Mosaico con 3 imagenes')

#Vemos que hace el mosaico correctamente







"""
4. (2.5 puntos) Lo mismo que en el punto anterior pero para N > 5 (usar las
imágenes para mosaico).
"""
#####################################################################################
#####################################################################################
                            # FUNCIONES EJERCICO 4 #
#####################################################################################
#####################################################################################




# Vamos a hacer una generalización del ejercicio 3. Para ello necesitamos una
# función que genere un mosaico a partir de dos imágenes:


# Método que genera un mosaico entre dos imágenes
################                 MOSAIC               ##############################   
####################################################################################  
def mosaic( img1, img2, ratio=0.75, reprojThresh=4.0 ):
    

    
    sift = cv.xfeatures2d.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des2,des1, k=2)
    
    
    good = []
    for m in matches:
        if m[0].distance < ratio*m[1].distance:
                good.append(m)
    matches = np.asarray(good)
    
    
   
    src = np.float32([ kp2[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
    dst = np.float32([ kp1[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)

    (H, status) = cv.findHomography(src, dst, cv.RANSAC, 1.0)
    
    
    
    dst = cv.warpPerspective(img2,H,(img1.shape[1] + img2.shape[1], img1.shape[0]),borderMode=cv.BORDER_TRANSPARENT)
    dst[0:img1.shape[0], 0:img1.shape[1]] = img1
    



    return dst


####################################################################################  
####################################################################################  



# Método que genera un mosaico entre N imágenes
################                 MOSAICN              ##############################   
#################################################################################### 
def mosaicN( img, ratio=0.75, reprojThresh=4.0 ):
    temp = img[len(img)-1]
    for i in range(len(img)-1,0,-1): 
        print(i)
        temp = mosaic(img[i-1],temp)


    return temp
####################################################################################  
####################################################################################  



# Vamos a componer el mosaico
res = mosaicN((img1,img2,img3,img4,img5,img6,img7,img8,img9,img10))
imShowRealScale(res,title='Mosaico con todas las imagenes')

# Vemos que aunque lo genera correctamente, pierde mucha calidad en los primeros mosaicos. 
# No he conseguido entender  por qué.


























#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: juliofg17
"""

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

## Función para mostrar imágenes
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




"""
Función para hacer unpack de cada octava. Fuente:
https://stackoverflow.com/questions/48385672/opencv-python-unpack-sift-octave
"""    

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










"""
SIFT
"""
####################################################################################
####################################################################################
def detectKeyPointsSIFT( img, sift ):

    kp_sift = sift.detect(img,None)
    imgkp = img.copy()
    imgkp = cv.drawKeypoints(imgkp,kp_sift,imgkp)
    
    
    
    
    
    unpacked_kp_sift = []
    for i in range(0,len(kp_sift)):
        unpacked_kp_sift.append(unpackSIFTOctave(kp_sift[i]))
        
    
    ## Num de octavas y capas
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
    
    
    ### listas para distintas octavas
    lista_octavas = []
    for i in range(num_oct):
        temp = []
        lista_octavas.append(temp)
    
    
    for i in range(len(unpacked_kp_sift)):
        #print(str(unpacked_kp[i][0]) + " " + str(min_oct) + " " + str(unpacked_kp[i][0]-min_oct))
        lista_octavas[unpacked_kp_sift[i][0]-min_oct].append(kp_sift[i])
        
        
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
      
    
    
   
    
    
    
    
    ### listas para distintas capas
    lista_capas = []
    for i in range(num_lay):
        temp = []
        lista_capas.append(temp)
    
    
    for i in range(len(unpacked_kp_sift)):
        #print(str(unpacked_kp[i][0]) + " " + str(min_oct) + " " + str(unpacked_kp[i][0]-min_oct))
        lista_capas[unpacked_kp_sift[i][1]-min_lay].append(kp_sift[i])
        
        
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
      
    
    
    return (imgkp, img_oct, img_lay, len(kp_sift))
####################################################################################
####################################################################################    
    

img1 = cv.imread('imagenes/Yosemite1.jpg')
sift1 = cv.xfeatures2d.SIFT_create(contrastThreshold=0.06,edgeThreshold=6)

(imgkp, img_oct, img_lay, num_kp) = detectKeyPointsSIFT( img1, sift1 )

imShowRealScale(imgkp)
imShowRealScale(img_oct)
imShowRealScale(img_lay)
print("Numero de Keypoints: " + str(num_kp))

img2 = cv.imread('imagenes/Yosemite2.jpg')
sift2 = cv.xfeatures2d.SIFT_create(contrastThreshold=0.06,edgeThreshold=6)

(imgkp, img_oct, img_lay, num_kp) = detectKeyPointsSIFT( img2, sift2 )

imShowRealScale(imgkp)
imShowRealScale(img_oct)
imShowRealScale(img_lay)
print("Numero de Keypoints: " + str(num_kp))






















"""
SURF
"""
####################################################################################
####################################################################################  
def detectKeyPointsSURF( img, surf ):

    kp_surf = surf.detect(img,None)
    
    imgkp = img.copy()
    imgkp = cv.drawKeypoints(imgkp,kp_surf,imgkp)
    
    unpacked_kp_surf = []
    for i in range(0,len(kp_surf)):
        unpacked_kp_surf.append(unpackSIFTOctave(kp_surf[i]))
    
    ## Num de octavas
    min_oct = unpacked_kp_surf[0][0]
    max_oct = unpacked_kp_surf[0][0]
    
 
    for i in range(len(unpacked_kp_surf)):
        if( unpacked_kp_surf[i][0] > max_oct ):
            max_oct = unpacked_kp_surf[i][0]
        if( unpacked_kp_surf[i][0] < min_oct ):
            min_oct = unpacked_kp_surf[i][0]

            
            
    num_oct = max_oct - min_oct +1

    
    
    ### listas para distintas octavas
    lista_octavas = []
    for i in range(num_oct):
        temp = []
        lista_octavas.append(temp)
    
    
    for i in range(len(unpacked_kp_surf)):
        #print(str(unpacked_kp[i][0]) + " " + str(min_oct) + " " + str(unpacked_kp[i][0]-min_oct))
        lista_octavas[unpacked_kp_surf[i][0]-min_oct].append(kp_surf[i])
        
        
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
      
    
    return (imgkp, img_oct, len(kp_surf))
    
####################################################################################
####################################################################################  


surf1 = cv.xfeatures2d.SURF_create(700)

(imgkp, img_oct, num_kp) = detectKeyPointsSURF( img1, surf1 )

imShowRealScale(imgkp)
imShowRealScale(img_oct)
print("Numero de Keypoints: " + str(num_kp))


surf2 = cv.xfeatures2d.SURF_create(600)

(imgkp, img_oct, num_kp) = detectKeyPointsSURF( img2, surf2 )

imShowRealScale(imgkp)
imShowRealScale(img_oct)
print("Numero de Keypoints: " + str(num_kp))






"""
Obtenemos descriptores
"""


(kp_sift1, desc_sift1) = sift1.detectAndCompute(img1,None)
(kp_surf1, desc_surf1) = surf1.detectAndCompute(img1,None)

(kp_sift2, desc_sift2) = sift2.detectAndCompute(img2,None)
(kp_surf2, desc_surf2) = surf2.detectAndCompute(img2,None)




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





























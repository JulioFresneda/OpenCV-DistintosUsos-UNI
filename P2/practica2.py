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
    

    
    ## Num de octavas
    min_oct = kp_surf[0].octave
    max_oct = kp_surf[0].octave
    
 
    for i in range(len(kp_surf)):
        if( kp_surf[i].octave > max_oct ):
            max_oct = kp_surf[i].octave
        if( kp_surf[i].octave < min_oct ):
            min_oct = kp_surf[0].octave

            
            
    num_oct = max_oct - min_oct +1

    
    
    ### listas para distintas octavas
    lista_octavas = []
    for i in range(num_oct):
        temp = []
        lista_octavas.append(temp)
    
    
    for i in range(len(kp_surf)):
      
        lista_octavas[kp_surf[i].octave-min_oct].append(kp_surf[i])
        
        
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
################      BRUTAL FORCE WITH CROSSCHECK    ##############################   
####################################################################################  
def detectMatchesBF( img1, img2, sift1, sift2, num_matches ):
    
    kp1, desc1 = sift1.detectAndCompute(img1,None)
    kp2, desc2 = sift2.detectAndCompute(img2,None)
    
    bf = cv.BFMatcher(crossCheck = True)
    matches = bf.match(desc1,desc2)
    
    random_matches = []
    from random import randint
    for i in range(num_matches):
        random_matches.append(matches[randint(0,len(matches)-1)])
    
    
    return (cv.drawMatches(img1,kp1,img2,kp2,random_matches,None), matches)
####################################################################################
#################################################################################### 
    

################         LOWE-AVERAGE-2NN             ##############################   
####################################################################################  
def detectMatchesKNN( img1, img2, sift1 = cv.xfeatures2d.SIFT_create(), sift2 = cv.xfeatures2d.SIFT_create(), num_matches = 100, ratio = 0.75, knn = 2 ):
    
    kp1, desc1 = sift1.detectAndCompute(img1,None)
    kp2, desc2 = sift2.detectAndCompute(img2,None)
    
    bf = cv.BFMatcher()
    matches = bf.knnMatch(desc1,desc2, k= knn)
    
    
    good_matches = []
    for m in matches:
        if m[0].distance < ratio*m[1].distance:
            good_matches.append(m)
    
    good_matches = np.asarray(good_matches)
    
    random_matches = []
    from random import randint
    for i in range(num_matches):
        random_matches.append(good_matches[randint(0,len(good_matches)-1)])
    
    
    return (cv.drawMatchesKnn(img1,kp1,img2,kp2,random_matches,None), good_matches)
####################################################################################
#################################################################################### 



"""
(a) Mostrar ambas imágenes en un mismo canvas y pintar lı́neas de difer-
entes colores entre las coordenadas de los puntos en correspondencias.
Mostrar en cada caso 100 elegidas aleatoriamente.
"""

img_matches_bf, matches_bf = detectMatchesBF(img1,img2,sift1,sift2,100)
imShowRealScale(img_matches_bf)

img_matches_knn, matches_knn = detectMatchesKNN(img1,img2,sift1,sift2,100)
imShowRealScale(img_matches_knn)



"""
(b) Valorar la calidad de los resultados obtenidos en términos de las corre-
spondencias válidas observadas por inspección ocular y las tendencias
de las lı́neas dibujadas.
"""



"""
(c) Comparar ambas técnicas de correspondencias en términos de la cal-
idad de sus correspondencias (suponer 100 aleatorias e inspección
visual).
"""







"""
3. (2.5 puntos) Escribir una función que genere un mosaico de calidad a
partir de N = 3 imágenes relacionadas por homografı́as, sus listas de
keyPoints calculados de acuerdo al punto anterior y las correspondencias
encontradas entre dichas listas. Estimar las homografı́as entre ellas usando
la función cv2.findHomography(p1,p2, CV RANSAC,1). Para el mosaico
será necesario.

"""




################                 MOSAIC               ##############################   
####################################################################################  

def mosaic( img1, img2, ratio=0.75, reprojThresh=4.0 ):
    

    
    sift = cv.xfeatures2d.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des2,des1, k=2)
    
    
    # Apply ratio test
    good = []
    for m in matches:
        if m[0].distance < 0.5*m[1].distance:
                good.append(m)
    matches = np.asarray(good)
    
    
   
    src = np.float32([ kp2[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
    dst = np.float32([ kp1[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)

    (H, status) = cv.findHomography(src, dst, cv.RANSAC, 1.0)
    
    
    
    dst = cv.warpPerspective(img2,H,(img1.shape[1] + img2.shape[1], img1.shape[0]))
    dst[0:img1.shape[0], 0:img1.shape[1]] = img1
    



    return dst


    
    
    
def mosaic3Img( img1, img2, img3, ratio=0.75, reprojThresh=4.0 ):
    temp = mosaic(img2,img3)
    
    imShowRealScale(temp)
    res = mosaic(img1,temp)
    return res





def mosaicNImg( img, ratio=0.75, reprojThresh=4.0 ):
    temp = img[len(img)-1]
    for i in range(len(img)-1,0,-1): 
        print(i)
        temp = mosaic(img[i-1],temp)
        imShowRealScale(temp)

    return temp


img1 = cv.imread('imagenes/mosaico002.jpg')
img2 = cv.imread('imagenes/mosaico003.jpg')
img3 = cv.imread('imagenes/mosaico004.jpg')
img4 = cv.imread('imagenes/mosaico005.jpg')
img5 = cv.imread('imagenes/mosaico006.jpg')


res = mosaicNImg((img1,img2,img3,img4,img5))

imShowRealScale(res)




























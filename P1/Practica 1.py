# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 20:19:52 2018

@author: Julio
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

"""
1.- USANDO LAS FUNCIONES DE OPENCV : 
escribir funciones que implementen los siguientes puntos: ( 2 puntos)


A) El cálculo de la convolución de una imagen con una máscara Gaussiana 2D 
(Usar GaussianBlur). Mostrar ejemplos con distintos tamaños de máscara y valores de sigma.
Valorar los resultados.
"""

# Imagen
img = cv2.imread('imagenes/cat.bmp')





# Aplicar a la imagen un kernel gaussiano para obtener distintas imágenes
difuminada1 = cv2.GaussianBlur(img,(5,5),0)
difuminada2 = cv2.GaussianBlur(img,(7,7),10)
difuminada3 = cv2.GaussianBlur(img,(3,3),0)

# OpenCV usa BGR, y matplotlib RGB. Por lo tanto hay que transformar
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
difuminada1 = cv2.cvtColor(difuminada1, cv2.COLOR_BGR2RGB)
difuminada2 = cv2.cvtColor(difuminada2, cv2.COLOR_BGR2RGB)
difuminada3 = cv2.cvtColor(difuminada3, cv2.COLOR_BGR2RGB)

# Mostrar imágenes
plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])

plt.subplot(122),plt.imshow(difuminada1),plt.title('Difuminada1')
plt.xticks([]), plt.yticks([])
plt.show()

plt.subplot(121),plt.imshow(difuminada2),plt.title('Difuminada2')
plt.xticks([]), plt.yticks([])

plt.subplot(122),plt.imshow(difuminada3),plt.title('Difuminada3')
plt.xticks([]), plt.yticks([])
plt.show()






"""
B) Usar getDerivKernels para obtener las máscaras 1D que permiten calcular la convolución 2D
con máscaras de derivadas. Representar e interpretar dichas máscaras 1D para distintos
valores de sigma.
"""

# Imagen
img = cv2.imread('imagenes/cat.bmp')


## Máscara derivada respecto a 'x' con sigma = 5
deriv5x = cv2.getDerivKernels(1, 0, 5, True)

## Máscara derivada respecto a 'y' con sigma = 5
deriv5y = cv2.getDerivKernels(0, 1, 5, True)

## Máscara derivada respecto a 'x' con sigma = 7
deriv7x = cv2.getDerivKernels(1, 0, 7, True)

## Máscara derivada respecto a 'y' con sigma = 7
deriv7y = cv2.getDerivKernels(0, 1, 7, True)





"""
C) Usar la función Laplacian para el cálculo de la convolución 2D con una máscara de
Laplaciana-de-Gaussiana de tamaño variable. Mostrar ejemplos de funcionamiento usando dos
tipos de bordes y dos valores de sigma: 1 y 3.
"""

# Imagen
img = cv2.imread('imagenes/cat.bmp')

## Laplacianas
lap1 = cv2.Laplacian(img,-1,1,1,0,cv2.BORDER_WRAP)
lap2 = cv2.Laplacian(img,-1,3,1,0,cv2.BORDER_WRAP)
lap3 = cv2.Laplacian(img,-1,1,1,0,cv2.BORDER_REFLECT)
lap4 = cv2.Laplacian(img,-1,3,1,0,cv2.BORDER_REFLECT)



# OpenCV usa BGR, y matplotlib RGB. Por lo tanto hay que transformar
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
lap1 = cv2.cvtColor(lap1, cv2.COLOR_BGR2RGB)
lap2 = cv2.cvtColor(lap2, cv2.COLOR_BGR2RGB)
lap3 = cv2.cvtColor(lap3, cv2.COLOR_BGR2RGB)
lap4 = cv2.cvtColor(lap4, cv2.COLOR_BGR2RGB)

# Mostrar imágenes
plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.show()

plt.subplot(121),plt.imshow(lap1),plt.title('Sigma=1, border wrap')
plt.xticks([]), plt.yticks([])

plt.subplot(122),plt.imshow(lap2),plt.title('Sigma=3, border wrap')
plt.xticks([]), plt.yticks([])
plt.show()

plt.subplot(121),plt.imshow(lap3),plt.title('Sigma=1, border reflect')
plt.xticks([]), plt.yticks([])


plt.subplot(122),plt.imshow(lap4),plt.title('Sigma=3, border wrap')
plt.xticks([]), plt.yticks([])
plt.show()











"""

2.- IMPLEMENTAR apoyándose en las funciones getDerivKernels, getGaussianKernel, 
pyrUp(), pyrDown(), escribir funciones los siguientes (3 puntos).

Usar solo imágenes de un solo canal (imágenes de gris). 
Valorar la influencia del tamaño de la máscara y el valor de sigma sobre la salida, 
en todos los casos.


A. El cálculo de la convolución 2D con una máscara separable de tamaño variable. 
Usar bordes reflejados. Mostrar resultados


"""

# Imagen
orig = cv2.imread('imagenes/cat.bmp',0)

def convolucion( orig, ksize, sigma ):
    kernel_array = cv2.getGaussianKernel(ksize,sigma)    
    kernel = []
    for k in range(0,len(kernel_array)):
        kernel.append(kernel_array[k][0])
    
    ancho_borde = ksize // 2
       
    
    
    ## AÑADIR BORDES
    img = []
    
    for row in range(0,len(orig)):
        row_izq = orig[row][:ancho_borde]
        row_dcho = orig[row][-ancho_borde:]
        
        
        row_izq = row_izq[::-1]
        row_dcho = row_dcho[::-1]
        
        concat = np.concatenate((row_izq,orig[row],row_dcho),axis=None)
        img.append(concat)
        
        
        
        
    column_up = img[:ancho_borde]
    column_bottom = img[-ancho_borde:]
    
    column_up = column_up[::-1]
    column_bottom = column_bottom[::-1]
    img = column_up + img + column_bottom
    

    ## CONVOLUCION
    
        ### Creamos una matriz vacía de 0
    n = len(img)-2*ancho_borde
    m = len(img[0])-2*ancho_borde
    conv = [0] * n
    for i in range(n):
        conv[i] = [0] * m

    
        ### HORIZONTALMENTE
    for row in range(0,len(conv)):
        for col in range(0,len(conv[0])):
            value = 0
            for k in range(0,len(kernel)):
                value = value + img[row+ancho_borde][col+k]*kernel[k]
                    
            conv[row][col] = value
            
            
    
        ## AÑADIR BORDES A LA CONV HORIZONTAL
    for row in range(0,len(conv)):
        row_izq = conv[row][:ancho_borde]
        row_dcho = conv[row][-ancho_borde:]
        
        row_izq = row_izq[::-1]
        row_dcho = row_dcho[::-1]
        
        conv[row] = row_izq + conv[row] + row_dcho
        
        
    column_up = conv[:ancho_borde]
    column_bottom = conv[-ancho_borde:]
    
    column_up = column_up[::-1]
    column_bottom = column_bottom[::-1]
    conv = column_up + conv + column_bottom
            
            
        ### Creamos otra matriz vacía de 0
    n = len(img)-2*ancho_borde
    m = len(img[0])-2*ancho_borde
    convfinal = [0] * n
    for i in range(n):
        convfinal[i] = [0] * m    
    
    
        ### VERTICALMENTE
    for col in range(0,len(convfinal[0])):
        for row in range(0,len(convfinal)):
            value = 0
            for k in range(0,len(kernel)):
                value = value + conv[row+k][col+ancho_borde]*kernel[k]
                    
            convfinal[row][col] = value
            
            
    
    return convfinal
    


### MOSTRAR IMAGEN

imgconv = convolucion(orig,5,3)

plt.subplot(121),plt.imshow(orig,cmap='gray'),plt.title('Original')
plt.xticks([]), plt.yticks([])

plt.subplot(122),plt.imshow(imgconv,cmap='gray'),plt.title('Convolucionada')
plt.xticks([]), plt.yticks([])
plt.show()



## Comparamos con la convolucion usando la función de opencv
difuminada1 = cv2.GaussianBlur(orig,(5,5),3)
plt.subplot(121),plt.imshow(difuminada1,cmap='gray'),plt.title('Convolucionada con función')
plt.xticks([]), plt.yticks([])
plt.show()






"""
B. El cálculo de la convolución 2D con una máscara 2D de 1ª derivada de tamaño variable. 
Mostrar ejemplos de funcionamiento usando bordes a cero.
"""








      
    







orig = [[0,2,4,6,8],[3,2,1,3,2],[4,4,0,2,4]]



ancho_borde = ksize // 2
    
    
    





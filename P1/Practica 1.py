# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 20:19:52 2018

@author: Julio Fresneda - juliofresnedag@correo.ugr.es
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

# Cargamos la imagen
img = cv2.imread('imagenes/cat.bmp')


# Aplicamos a la imagen un kernel Gaussiano para obtener distintas imágenes difuminadas, con distintos tamaños
# de máscara y valores de sigma.
difuminada1 = cv2.GaussianBlur(img,(5,5),1)
difuminada2 = cv2.GaussianBlur(img,(7,7),10)
difuminada3 = cv2.GaussianBlur(img,(3,3),5)

# OpenCV usa BGR, y matplotlib RGB. Por lo tanto, para visualizar bien las imágenes hay que transformarlas.
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
difuminada1 = cv2.cvtColor(difuminada1, cv2.COLOR_BGR2RGB)
difuminada2 = cv2.cvtColor(difuminada2, cv2.COLOR_BGR2RGB)
difuminada3 = cv2.cvtColor(difuminada3, cv2.COLOR_BGR2RGB)

# Mostramos las imágenes
plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])

plt.subplot(122),plt.imshow(difuminada1),plt.title('ksize 5x5, sigma = 1')
plt.xticks([]), plt.yticks([])
plt.show()

plt.subplot(121),plt.imshow(difuminada2),plt.title('ksize 7x7, sigma=10')
plt.xticks([]), plt.yticks([])

plt.subplot(122),plt.imshow(difuminada3),plt.title('ksize 3x3, sigma = 5')
plt.xticks([]), plt.yticks([])
plt.show()





"""
B) Usar getDerivKernels para obtener las máscaras 1D que permiten calcular la convolución 2D
con máscaras de derivadas. Representar e interpretar dichas máscaras 1D para distintos
valores de sigma.
"""

##### KÉRNELS


## Ksize = 1

## Máscara derivada respecto a 'x' 
d1x_x, d1x_y = cv2.getDerivKernels(1, 0, 1, True)

## Matriz:
d1x_x = d1x_x.transpose()
d1xm = d1x_x * d1x_y
d1xm

## Máscara derivada respecto a 'y' 
d1y_x, d1y_y = cv2.getDerivKernels(0, 1, 1, True)

## Matriz:
d1y_x = d1y_x.transpose()
d1ym = d1y_x * d1y_y
d1ym

## Máscara derivada respecto a 'x' e 'y'
d1xy_x, d1xy_y = cv2.getDerivKernels(1, 1, 1, True)

## Matriz:
d1xy_x = d1xy_x.transpose()
d1xym = d1xy_x * d1xy_y
d1xym


## Ksize = 3

## Máscara derivada respecto a 'x' 
d3x_x, d3x_y = cv2.getDerivKernels(1, 0, 3, True)

## Matriz:
d3x_x = d3x_x.transpose()
d3xm = d3x_x * d3x_y
d3xm

## Máscara derivada respecto a 'y' 
d3y_x, d3y_y = cv2.getDerivKernels(0, 1, 3, True)

## Matriz:
d3y_x = d3y_x.transpose()
d3ym = d3y_x * d3y_y
d3ym

## Máscara derivada respecto a 'x' e 'y'
d3xy_x, d3xy_y = cv2.getDerivKernels(1, 1, 3, True)

## Matriz:
d3xy_x = d3xy_x.transpose()
d3xym = d3xy_x * d3xy_y
d3xym



## Ksize = 5

## Máscara derivada respecto a 'x' 
d5x_x, d5x_y = cv2.getDerivKernels(1, 0, 5, True)

## Matriz:
d5x_x = d5x_x.transpose()
d5xm = d5x_x * d5x_y
d5xm

## Máscara derivada respecto a 'y' 
d5y_x, d5y_y = cv2.getDerivKernels(0, 1, 5, True)

## Matriz:
d5y_x = d5y_x.transpose()
d5ym = d5y_x * d5y_y
d5ym

## Máscara derivada respecto a 'x' e 'y'
d5xy_x, d5xy_y = cv2.getDerivKernels(1, 1, 5, True)

## Matriz:
d5xy_x = d5xy_x.transpose()
d5xym = d5xy_x * d5xy_y
d5xym



## Representación visual de kernel 3x3 respecto a la x
kernel3x = []
for k in range(0,len(d3x_x[0])):
    kernel3x.append(d3x_x[0][k])

kernel3y = []
for k in range(0,len(d3x_y)):
    kernel3y.append(d3x_y[k][0])
    
img_bordevertical = [[10,10,10,20,20,20],[10,10,10,20,20,20],[10,10,10,20,20,20],[10,10,10,20,20,20],[10,10,10,20,20,20],[10,10,10,20,20,20]]
c = convolucionKernelSeparable(img_bordevertical,kernel3x,kernel3y,0)

plt.subplot(121),plt.imshow(img_bordevertical),plt.title('Imagen con borde vertical')
plt.xticks([]), plt.yticks([])

plt.subplot(122),plt.imshow(c),plt.title('Borde detectado')
plt.xticks([]), plt.yticks([])
plt.show()





## Representación visual de aplicarle kernels 5x5 a una imagen con borde vertical y horizontal
imgconbordes = [[10,10,10,10,10,10,10,20,20,20],[10,10,10,10,10,10,10,20,20,20],[10,10,10,10,10,10,10,20,20,20],[10,10,10,10,10,10,10,20,20,20],[10,10,10,10,10,10,10,20,20,20],[10,10,10,10,10,10,10,20,20,20],[30,30,30,30,30,30,30,20,20,20],[30,30,30,30,30,30,30,20,20,20],[30,30,30,30,30,30,30,20,20,20]]


# Derivada respecto a la x
kernel5x = []
for k in range(0,len(d5x_x[0])):
    kernel5x.append(d5x_x[0][k])

kernel5y = []
for k in range(0,len(d5x_y)):
    kernel5y.append(d5x_y[k][0])
    
cx = convolucionKernelSeparable(imgconbordes,kernel5x,kernel5y,0)

# Derivada respecto a la y
kernel5x = []
for k in range(0,len(d5y_x[0])):
    kernel5x.append(d5y_x[0][k])

kernel5y = []
for k in range(0,len(d5y_y)):
    kernel5y.append(d5y_y[k][0])
    
cy = convolucionKernelSeparable(imgconbordes,kernel5x,kernel5y,0)
    
# Derivada respecto a la x y la y
kernel5x = []
for k in range(0,len(d5xy_x[0])):
    kernel5x.append(d5xy_x[0][k])

kernel5y = []
for k in range(0,len(d5xy_y)):
    kernel5y.append(d5xy_y[k][0])
    
cxy = convolucionKernelSeparable(imgconbordes,kernel5x,kernel5y,0)


plt.subplot(121),plt.imshow(imgconbordes),plt.title('Imagen con bordes')
plt.xticks([]), plt.yticks([])

plt.subplot(122),plt.imshow(cx),plt.title('Kernel de la deriv. a la x')
plt.xticks([]), plt.yticks([])
plt.show() 

plt.subplot(121),plt.imshow(cy),plt.title('Kernel de la deriv. a la y')
plt.xticks([]), plt.yticks([])

    
plt.subplot(122),plt.imshow(cxy),plt.title('Kernel de la deriv. a la x e y')
plt.xticks([]), plt.yticks([])
plt.show()  







"""
C) Usar la función Laplacian para el cálculo de la convolución 2D con una máscara de
Laplaciana-de-Gaussiana de tamaño variable. Mostrar ejemplos de funcionamiento usando dos
tipos de bordes y dos valores de sigma: 1 y 3.
"""

# Imagen
img = cv2.imread('imagenes/plane.bmp')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

## Vamos a pasarle un blur gaussiano para eliminar ruido
imgs1 = cv2.GaussianBlur(img,(0,0),1)
imgs3 = cv2.GaussianBlur(img,(0,0),3)

## Laplacianas

# Sigma = 1
laps1i = cv2.Laplacian(imgs1,cv2.CV_8U,borderType = cv2.BORDER_ISOLATED,delta=100)
laps1r = cv2.Laplacian(imgs1,cv2.CV_8U,borderType = cv2.BORDER_REFLECT,delta=100)

# Sigma = 3
laps3i = cv2.Laplacian(imgs3,cv2.CV_8U,borderType = cv2.BORDER_ISOLATED,delta=100)
laps3r = cv2.Laplacian(imgs3,cv2.CV_8U,borderType = cv2.BORDER_REFLECT,delta=100)


# Mostrar imágenes
plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.show()

plt.subplot(121),plt.imshow(laps1i),plt.title('Sigma=1, border isolated')
plt.xticks([]), plt.yticks([])

plt.subplot(122),plt.imshow(laps1r),plt.title('Sigma=1, border reflect')
plt.xticks([]), plt.yticks([])
plt.show()

plt.subplot(121),plt.imshow(laps3i),plt.title('Sigma=3, border isolated')
plt.xticks([]), plt.yticks([])


plt.subplot(122),plt.imshow(laps3r),plt.title('Sigma=3, border reflect')
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

       
## BONUS 3
def convolucionKernelSeparable( orig, kernelx, kernely, tipo_borde ):
    
    
    ancho_borde = len(kernelx) // 2
       
    
    
    ## AÑADIR BORDES
    img = []
    if( tipo_borde == 0 ): ## Reflejados
    
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
    



    else:
        if( tipo_borde == 1 ): # Ceros
      
            row = []
            for r in range(0,ancho_borde):
                row.append(0)
            for r in range(0,len(orig)):
       
                concat = np.concatenate((row,orig[r],row),axis=None)
                img.append(concat)
                
                
            col = []
            for c in range(0,len(img[0])):
                col.append(0)
                
            allcol = []
            for x in range(0,ancho_borde):
                allcol.append(col)
        
            img = allcol + img + allcol


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
            for k in range(0,len(kernelx)):
                value = value + img[row+ancho_borde][col+k]*kernelx[k]
                    
            conv[row][col] = value
            
            
    
        ## AÑADIR BORDES A LA CONV HORIZONTAL
        
        
    if( tipo_borde == 0 ):
        
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
        
        
    else:
        if( tipo_borde == 1 ): # Ceros
      
            row = []
            for r in range(0,ancho_borde):
                row.append(0)
            for r in range(0,len(conv)):
       
                concat = np.concatenate((row,conv[r],row),axis=None)
                conv[r] = concat
                
                
                   
            col = []
            for c in range(0,len(conv[0])):
                col.append(0)
                
            allcol = []
            for x in range(0,ancho_borde):
                allcol.append(col)
        
            conv = allcol + conv + allcol

            
            
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
            for k in range(0,len(kernely)):
                value = value + conv[row+k][col+ancho_borde]*kernely[k]
                    
            convfinal[row][col] = value
            
            
    
    return convfinal
    


### MOSTRAR IMAGEN
    
# Imagen
orig = cv2.imread('imagenes/marilyn.bmp',0)


# Kernel con tamaño de máscara 3 y sigma 1
kernel_array = cv2.getGaussianKernel(3,1)    
kernel31 = []
for k in range(0,len(kernel_array)):
    kernel31.append(kernel_array[k][0])
    
# Kernel con tamaño de máscara 5 y sigma 1
kernel_array = cv2.getGaussianKernel(5,1)    
kernel51 = []
for k in range(0,len(kernel_array)):
    kernel51.append(kernel_array[k][0])
    
    
# Kernel con tamaño de máscara 3 y sigma 3
kernel_array = cv2.getGaussianKernel(3,3)    
kernel33 = []
for k in range(0,len(kernel_array)):
    kernel33.append(kernel_array[k][0])
    
    
    # Kernel con tamaño de máscara 5 y sigma 3
kernel_array = cv2.getGaussianKernel(5,3)    
kernel53 = []
for k in range(0,len(kernel_array)):
    kernel53.append(kernel_array[k][0])
    
    
## Convolucionamos
imgconv31 = convolucionKernelSeparable(orig,kernel31,kernel31,0)
imgconv33 = convolucionKernelSeparable(orig,kernel33,kernel33,0)
imgconv51 = convolucionKernelSeparable(orig,kernel51,kernel51,0)
imgconv53 = convolucionKernelSeparable(orig,kernel53,kernel53,0)


## Mostramos las imágenes
plt.subplot(121),plt.imshow(orig,cmap='gray'),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.show()

plt.subplot(121),plt.imshow(imgconv31,cmap='gray'),plt.title('Conv ksize=3 sigma=1')
plt.xticks([]), plt.yticks([])

plt.subplot(122),plt.imshow(imgconv33,cmap='gray'),plt.title('Conv ksize=3 sigma=3')
plt.xticks([]), plt.yticks([])
plt.show()

plt.subplot(121),plt.imshow(imgconv51,cmap='gray'),plt.title('Conv ksize=5 sigma=1')
plt.xticks([]), plt.yticks([])

plt.subplot(122),plt.imshow(imgconv53,cmap='gray'),plt.title('Conv ksize=5 sigma=3')
plt.xticks([]), plt.yticks([])
plt.show()




## Comparamos esta última con la convolucion usando la función de opencv
difuminada1 = cv2.GaussianBlur(orig,(5,5),3)
plt.subplot(121),plt.imshow(difuminada1,cmap='gray'),plt.title('Convolucionada con función')
plt.xticks([]), plt.yticks([])
plt.show()





"""
B. El cálculo de la convolución 2D con una máscara 2D de 1ª derivada de tamaño variable. 
Mostrar ejemplos de funcionamiento usando bordes a cero.
"""


## Obtenemos los coeficientes del kernel
kernel_array3 = cv2.getDerivKernels(1,1,3)

kernelx3 = []
for k in range(0,len(kernel_array3[0])):
    kernelx3.append(kernel_array3[0][k][0])
    
kernely3 = []
for k in range(0,len(kernel_array3[1])):
    kernely3.append(kernel_array3[1][k][0])
    
    
kernel_array7 = cv2.getDerivKernels(1,1,7)

kernelx7 = []
for k in range(0,len(kernel_array7[0])):
    kernelx7.append(kernel_array7[0][k][0])
    
kernely7 = []
for k in range(0,len(kernel_array7[1])):
    kernely7.append(kernel_array7[1][k][0])


## Cargamos la imagen
orig = cv2.imread('imagenes/bird.bmp',0)

## Le pasamos un blur para eliminar ruido
img = cv2.GaussianBlur(orig,(5,5),3)


## Obtenemos la convolución
imgconvk3 = convolucionKernelSeparable(img,kernelx3,kernely3,1)
imgconvk7 = convolucionKernelSeparable(img,kernelx7,kernely7,1)

plt.subplot(121),plt.imshow(orig,cmap='gray'),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.show()


plt.subplot(121),plt.imshow(imgconvk3,cmap='gray'),plt.title('Convulcionada k=3')
plt.xticks([]), plt.yticks([])

plt.subplot(122),plt.imshow(imgconvk7,cmap='gray'),plt.title('Convolucionada k=7')
plt.xticks([]), plt.yticks([])
plt.show()




""" 
C. El cálculo de la convolución 2D con una máscara 2D de 2ª derivada de tamaño variable.

"""



## Obtenemos los coeficientes para la derivada
kernel_array3 = cv2.getDerivKernels(2,2,3)

kernelx3 = []
for k in range(0,len(kernel_array3[0])):
    kernelx3.append(kernel_array3[0][k][0])
    
kernely3 = []
for k in range(0,len(kernel_array3[1])):
    kernely3.append(kernel_array3[1][k][0])
    
    
kernel_array7 = cv2.getDerivKernels(2,2,7)

kernelx7 = []
for k in range(0,len(kernel_array7[0])):
    kernelx7.append(kernel_array7[0][k][0])
    
kernely7 = []
for k in range(0,len(kernel_array7[1])):
    kernely7.append(kernel_array7[1][k][0])


## Cargamos la imagen
orig = cv2.imread('imagenes/bird.bmp',0)

## Le pasamos un blur para eliminar ruido
img = cv2.GaussianBlur(orig,(5,5),1)


## Obtenemos la convolución
imgconvk3 = convolucionKernelSeparable(img,kernelx3,kernely3,1)
imgconvk7 = convolucionKernelSeparable(img,kernelx7,kernely7,1)

plt.subplot(121),plt.imshow(orig,cmap='gray'),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.show()


plt.subplot(121),plt.imshow(imgconvk3,cmap='gray'),plt.title('Convulcionada k=3')
plt.xticks([]), plt.yticks([])

plt.subplot(122),plt.imshow(imgconvk7,cmap='gray'),plt.title('Convolucionada k=7')
plt.xticks([]), plt.yticks([])
plt.show()




"""
D. Una función que genere una representación en pirámide Gaussiana de 4 niveles de una imagen.
Mostrar ejemplos de funcionamiento usando bordes.
"""
## Función para mostrar imágenes

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



## Función
def gaussianPyramid(img,levels,border = 4):
    

    layer = img
    
    gaussian_pyramid = [layer]
    for i in range(levels):
        layer = cv2.pyrDown(layer,borderType = border)
        gaussian_pyramid.append(layer)
    

    return gaussian_pyramid



# Cargamos la imagen
orig = cv2.imread('imagenes/cat.bmp',0)


## Probamos la pirámide con bordes reflejados

gp = gaussianPyramid(orig,4,2)

print("Bordes reflejados:")
imShowRealScale(gp[0],1,'Original')
imShowRealScale(gp[1],1,'Gaussian nivel 1')
imShowRealScale(gp[2],1,'Gaussian nivel 2')
imShowRealScale(gp[3],1,'Gaussian nivel 3')
imShowRealScale(gp[4],1,'Gaussian nivel 4')



## Probamos la pirámide con bordes wrap

gp = gaussianPyramid(orig,4,3)


print("Bordes wrap:")
imShowRealScale(gp[0],1,'Original')
imShowRealScale(gp[1],1,'Gaussian nivel 1')
imShowRealScale(gp[2],1,'Gaussian nivel 2')
imShowRealScale(gp[3],1,'Gaussian nivel 3')
imShowRealScale(gp[4],1,'Gaussian nivel 4')




"""
E. Una función que genere una representación en pirámide Laplaciana de 4 niveles de una imagen.
Mostrar ejemplos de funcionamiento usando bordes.

"""


def laplacianPyramid(levels,gp):

    laplacian_pyramid = []
    for i in range(levels,0,-1):
        print(i)
        high_f = gp[i-1]
        size = (high_f.shape[1], high_f.shape[0])
        
        expanded = cv2.pyrUp(gp[i],dstsize=size)

        laplacian_pyramid.append(cv2.subtract(high_f,expanded)) 
        
    return list(reversed(laplacian_pyramid))
        
        
        
## Probamos la pirámide
orig = cv2.imread('imagenes/einstein.bmp',0)
gp = gaussianPyramid(orig,4)
lp = laplacianPyramid(4,gp)

imShowRealScale(gp[0],1,'Gaussian nivel 0')
imShowRealScale(gp[1],1,'Gaussian nivel 1')
imShowRealScale(gp[2],1,'Gaussian nivel 2')
imShowRealScale(gp[3],1,'Gaussian nivel 3')

imShowRealScale(lp[0],1,'Laplacian nivel 0')
imShowRealScale(lp[1],1,'Laplacian nivel 1')
imShowRealScale(lp[2],1,'Laplacian nivel 2')
imShowRealScale(lp[3],1,'Laplacian nivel 3')





"""

3.- Imágenes Híbridas: (SIGGRAPH 2006 paper by Oliva, Torralba, and Schyns). (3 puntos)
Mezclando adecuadamente una parte de las frecuencias altas de una imagen con una parte de las frecuencias bajas de otra
imagen, obtenemos una imagen híbrida que admite distintas interpretaciones a distintas distancias ( ver hybrid images
project page).Para seleccionar la parte de frecuencias altas y bajas que nos quedamosde cada una de las imágenes
usaremos el parámetro sigma del núcleo/máscara de alisamiento gaussiano que usaremos. A mayor valor de sigma mayor
eliminación de altas frecuencias en la imagen convolucionada. Para una buena implementación elegir dicho valor de
forma separada para cada una de las dos imágenes ( ver las recomendaciones dadas en el paper de Oliva et al.). 
Recordar que las máscaras 1D siempre deben tener de longitud un número impar. Implementar una función que genere las
imágenes de baja y alta frecuencia a partir de las parejas de imágenes ( solo en la versión de imágenes de gris) .
El valor de sigma más adecuado para cada pareja habrá que encontrarlo por experimentación.
    1. Escribir una función que muestre las tres imágenes ( alta, baja e híbrida) en una misma ventana. 
  (Recordar que las imágenes después de una convolución contienen número flotantes que pueden ser positivos y negativos)
    2. Realizar la composición con al menos 3 de las parejas de imágenes.
    
    
"""


## Función que obtiene la imagen híbrida
def hibridas(img_baja, sigma_baja, img_alta, sigma_alta,g=0):
   
    img_lejos = cv2.GaussianBlur(img_baja,(0,0),sigma_baja)
    gau = cv2.GaussianBlur(img_alta,(5,5),sigma_alta)
  
    img_cerca = cv2.subtract(img_alta,gau)
    
    hibrida = cv2.add(img_lejos,img_cerca)
    tit = 'Frecuencias bajas, sigma = ' + str(sigma_baja)
    imShowRealScale(img_lejos,gray=g,title=tit )
    tit = 'Frecuencias altas, sigma = ' + str(sigma_alta)
    imShowRealScale(img_cerca,gray=g,title=tit)
    imShowRealScale(hibrida,gray=g,title='Híbrida')
    imShowRealScale(hibrida,scale=0.25,gray=g,title='Híbrida')
    
    return hibrida
    
    
# Pez y submarino
lejos = cv2.imread('imagenes/submarine.bmp',0)
cerca = cv2.imread('imagenes/fish.bmp',0)

pezsub = hibridas(lejos,5,cerca,3.5,1)


# Avión y pájaro
lejos = cv2.imread('imagenes/bird.bmp',0)
cerca = cv2.imread('imagenes/plane.bmp',0)

avpaj = hibridas(lejos,5,cerca,2,1)


# Bici y moto
lejos = cv2.imread('imagenes/motorcycle.bmp',0)
cerca = cv2.imread('imagenes/bicycle.bmp',0)

bicimoto = hibridas(lejos,5,cerca,1,1)








"""
BONUS
"""


"""
3.- Implementar con código propio la convolución 2D con cualquier máscara 2D de números reales usando máscaras separables. (2 puntos)
"""

## Ya hecho en ejercicios anteriores

"""
3.- Construir una pirámide Gaussiana de al menos 5 niveles con las imágenes híbridas calculadas en el apartado
anterior. Mostrar los distintos niveles de la pirámide en un único canvas e interpretar el resultado.
Usar implementaciones propias de todas las funciones usadas (0.5 puntos)
"""

gp = gaussianPyramid(avpaj,5,2)

imShowRealScale(gp[0],1,'Original, nivel 1')
imShowRealScale(gp[1],1,'Gaussian nivel 2')
imShowRealScale(gp[2],1,'Gaussian nivel 3')
imShowRealScale(gp[3],1,'Gaussian nivel 4')
imShowRealScale(gp[4],1,'Gaussian nivel 5')

gp = gaussianPyramid(pezsub,5,2)

imShowRealScale(gp[0],1,'Original, nivel 1')
imShowRealScale(gp[1],1,'Gaussian nivel 2')
imShowRealScale(gp[2],1,'Gaussian nivel 3')
imShowRealScale(gp[3],1,'Gaussian nivel 4')
imShowRealScale(gp[4],1,'Gaussian nivel 5')

gp = gaussianPyramid(bicimoto,5,2)

imShowRealScale(gp[0],1,'Original, nivel 1')
imShowRealScale(gp[1],1,'Gaussian nivel 2')
imShowRealScale(gp[2],1,'Gaussian nivel 3')
imShowRealScale(gp[3],1,'Gaussian nivel 4')
imShowRealScale(gp[4],1,'Gaussian nivel 5')



"""
4.- Realizar todas las parejas de imágenes híbridas en su formato a color (1 punto) 
(solo se tendrá en cuenta si la versión de gris es correcta)
"""

# Pez y submarino
lejos = cv2.imread('imagenes/submarine.bmp')
cerca = cv2.imread('imagenes/fish.bmp')

lejos = cv2.cvtColor(lejos, cv2.COLOR_BGR2RGB)
cerca = cv2.cvtColor(cerca, cv2.COLOR_BGR2RGB)

pezsub = hibridas(lejos,5,cerca,3.5)


# Avión y pájaro
lejos = cv2.imread('imagenes/bird.bmp')
cerca = cv2.imread('imagenes/plane.bmp')

lejos = cv2.cvtColor(lejos, cv2.COLOR_BGR2RGB)
cerca = cv2.cvtColor(cerca, cv2.COLOR_BGR2RGB)

avpaj = hibridas(lejos,5,cerca,2.5)


# Bici y moto
lejos = cv2.imread('imagenes/motorcycle.bmp')
cerca = cv2.imread('imagenes/bicycle.bmp')

lejos = cv2.cvtColor(lejos, cv2.COLOR_BGR2RGB)
cerca = cv2.cvtColor(cerca, cv2.COLOR_BGR2RGB)

bicimoto = hibridas(lejos,5,cerca,1.5)


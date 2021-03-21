import cv2
import numpy as np

def nothing(x):
   pass

# criando a captura
videoIn = 0
cap = cv2.VideoCapture(videoIn)

# Criando uma janela para os controles
cv2.namedWindow('controles')
cv2.createTrackbar('Hue Minimo','controles',0,255,nothing)
cv2.createTrackbar('Hue Maximo','controles',0,255,nothing)
cv2.createTrackbar('Saturation Minimo','controles',0,255,nothing)
cv2.createTrackbar('Saturation Maximo','controles',0,255,nothing)
cv2.createTrackbar('Value Minimo','controles',0,255,nothing)
cv2.createTrackbar('Value Maximo','controles',0,255,nothing)

while True:

    # Lendo o frame da câmera
    _,cv_image = cap.read() 

    # Convertendo para HSV
    hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

    # Capturando o valor dos Trackbars
    hMin = cv2.getTrackbarPos('Hue Minimo','controles')
    hMax = cv2.getTrackbarPos('Hue Maximo','controles')
    sMin = cv2.getTrackbarPos('Saturation Minimo','controles')
    sMax = cv2.getTrackbarPos('Saturation Maximo','controles')
    vMin = cv2.getTrackbarPos('Value Minimo','controles')
    vMax = cv2.getTrackbarPos('Value Maximo','controles')

    #Se crea un array con las posiciones minimas y maximas
    lower=np.array([hMin,sMin,vMin])
    upper=np.array([hMax,sMax,vMax])

    #Deteccion de colores
    mask = cv2.inRange(hsv, lower, upper)

    #Mostrar los resultados y salir
    mask_show = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    output = np.hstack((cv_image, mask_show))
    cv2.imshow('controles',output)
    #cv2.imshow('mask',mask)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
  
arq = open('parametros.txt', 'w')
texto = []
texto.append('valorClaro = np.array([' + str(hMin) + ', ' + str(sMin) + ', ' + str(vMin) + '], dtype=np.uint8)\n')
texto.append('valorEscuro = np.array([' + str(hMax) + ', ' + str(sMax) + ', ' + str(vMax) + '], dtype=np.uint8)\n')
arq.writelines(texto)
arq.close()

cv2.destroyAllWindows()

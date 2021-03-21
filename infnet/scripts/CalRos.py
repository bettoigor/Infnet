#!/usr/bin/env python
# -*- coding: utf-8 -*-

import serial
import cv2, rospy
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image



cap = cv2.VideoCapture(0)
 
def nothing(x):
   pass

def callback_img (msg):
  
  global bridge
  global cv_image
  global hsv
  
  cv_image = bridge.imgmsg_to_cv2(msg,desired_encoding="bgr8")
  hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
  #cv2.imshow("HSV", cv_image)
  #cv2.waitKey(3) 


bridge = CvBridge()
cv_image = 0
hsv = 0
rospy.init_node('robot_image', anonymous=True)
rospy.Subscriber('/gpg/image', Image, callback_img)     # nome do node de publucação de imagem

#Creamos una ventana llamada 'image' en la que habra todos los sliders
cv2.namedWindow('controles')
cv2.createTrackbar('Hue Minimo','controles',0,255,nothing)
cv2.createTrackbar('Hue Maximo','controles',0,255,nothing)
cv2.createTrackbar('Saturation Minimo','controles',0,255,nothing)
cv2.createTrackbar('Saturation Maximo','controles',0,255,nothing)
cv2.createTrackbar('Value Minimo','controles',0,255,nothing)
cv2.createTrackbar('Value Maximo','controles',0,255,nothing)
rate = rospy.Rate(30) 
rospy.sleep(1)

while not rospy.is_shutdown():
  
  #_,frame = cap.read() #Leer un frame
  hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV) #Convertirlo a espacio de color HSV
  
  #Los valores maximo y minimo de H,S y V se guardan en funcion de la posicion de los sliders
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
  cv2.imshow('camara',cv_image)
  cv2.imshow('mask',mask)
  cv2.waitKey(5)
  '''
  k = cv2.waitKey(5) & 0xFF
  if k == 27:
    break
  '''
  rate.sleep()


arq = open('/home/pi/Documents/Testes/parametros.txt', 'w')
texto = []
texto.append('valorClaro = np.array([' + str(hMin) + ', ' + str(sMin) + ', ' + str(vMin) + '], dtype=np.uint8)\n')
texto.append('valorEscuro = np.array([' + str(hMax) + ', ' + str(sMax) + ', ' + str(vMax) + '], dtype=np.uint8)\n')
arq.writelines(texto)
arq.close()

cv2.destroyAllWindows()

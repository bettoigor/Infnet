import cv2
import numpy as np

# criando a captura
videoIn = 0
cap = cv2.VideoCapture(videoIn)

#cv_image = cv2.imread('fig.jpg')

while  True:

    _,cv_image = cap.read() 

    cv2.namedWindow('Original')
    cv2.namedWindow('Gray')
    gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    gray_rgb = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
    low = [4, 65, 84]
    high = [14, 255, 168]
    valorClaro = np.array(low, dtype=np.uint8)
    valorEscuro = np.array(high, dtype=np.uint8)


    mask = cv2.inRange(hsv, valorClaro, valorEscuro)
    mask = cv2.GaussianBlur(mask,(5,5),5)

    # applying morphological operations
    kernel = np.ones((3),np.uint8)

    mask = cv2.dilate(mask,kernel,iterations=2)
    mask = cv2.erode(mask,kernel,iterations=2)
    
    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN, kernel)


    out_image = cv2.bitwise_and(cv_image,cv_image,mask = mask)
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Fiding mask contours
    contour = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)

    cnt = contour[0]
    M = cv2.moments(cnt)

    x,y,w,h = cv2.boundingRect(cnt)

    cv2.rectangle(cv_image,(x,y),(x+w,y+h),(0,255,0),2)
    
    output = np.hstack((cv_image, mask_rgb, out_image))
    #out = np.hstack((output, mask_rgb))
    cv2.imshow('Original', cv_image)
    cv2.imshow('Gray',output)



    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        breakcv2.waitKey(1)
        break

cv2.destroyAllWindows()
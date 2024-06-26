import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
segmentor = SelfiSegmentation()
fpsReader = cvzone.FPS()
imgBg = cv2.imread("images/1.jpg")

while True:
    success, img = cap.read()
    imgOut = segmentor.removeBG(img,imgBg, threshold = 0.8) #threshold determines how accurately cut picture you want


    _,imgStacked = cvzone.stackImages([img, imgOut], 2,1)
    fpsReader.update(imgStacked, color=(0,0,255 ))
    cv2.imshow("Image", imgStacked)
    #cv2.imshow("Image Out", imgOut)
    cv2.waitKey(1)


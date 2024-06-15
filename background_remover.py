import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
segmentor = SelfiSegmentation()


while True:
    success, img = cap.read()
    imgOut = segmentor.removeBG(img,(255,0,255))

    imgStacked = cvzone.stackImages([img, imgOut], 2,1)
    cv2.imshow("Image", imgStacked)
    #cv2.imshow("Image Out", imgOut)
    cv2.waitKey(1)


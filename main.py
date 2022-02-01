import cv2
from tracker import *
import numpy as np

tracker = EuclideanDistTracker()
capture = cv2.VideoCapture("highway.mp4") #video path
delay = 30

object_dec = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40) #selects the
#appropriate number of gaussian distribution for each pixel, runs better than KNN




while 1 : #each loop = one frame
    ret, frame = capture.read()
    h,w,_ = frame.shape
    #print(h,w)
    #extract regions of interest
    region = frame[340:720,500:800] #focus on a specific road

    #obj detec
    mask = object_dec.apply(region)
    _, mask = cv2.threshold(mask, 254 ,255 ,cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    det = []
    for c in contours:
        #calculate area and remove small elements :
        area = cv2.contourArea(c)
        if area > 100:
            #cv2.drawContours(region, [c], -1, (0, 255, 0), 2)
            x,y,w,h = cv2.boundingRect(c)
            det.append([x,y,w,h])
    ids = tracker.update(det)
    for i in ids :
        x, y, w, h, id = i
        cv2.putText(region,str(id),(x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.rectangle(region, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv2.imshow("region",region)
    cv2.imshow("Mask", mask) #detected objects are white
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(delay) #waits for pressed key

    if key == 27: #esc on keyboard
        break


capture.release()
cv2.destroyWindow()





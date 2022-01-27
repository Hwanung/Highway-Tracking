import cv2


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
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for x in contours:
        #calculate area and remove small elements :
        area = cv2.contourArea(x)
        if area > 100:
            cv2.drawContours(region, [x], -1, (0, 255, 0), 2)
    cv2.imshow("region",region)
    cv2.imshow("Mask", mask) #detected objects are white
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(delay) #waits for pressed key

    if key == 27: #esc on keyboard
        break


capture.release()
cv2.destroyWindow()





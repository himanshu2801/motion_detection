import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

ret, frame1 = cap.read()
ret, frame2 = cap.read()
l=0
while cap.isOpened():
    difference = cv.absdiff(frame1, frame2)
    gray = cv.cvtColor(difference, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5,5), 0)
    _, threshold1 = cv.threshold(blurred, 20, 255, cv.THRESH_BINARY)
    dilate_image = cv.dilate(threshold1, None, iterations=3)
    contours, _ = cv.findContours(dilate_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for contour1 in contours:
        if cv.contourArea(contour1) > 5000:
            l=l+1
    for contour in contours:
        (x, y, width, hieght) = cv.boundingRect(contour)
        if cv.contourArea(contour) < 5000:
            continue
        cv.rectangle(frame1, (x, y), (x+width, y+hieght), (0,255,255), 3)
        cv.putText(frame1, "Count: {}".format(str(l)), (10,20), cv.FONT_ITALIC, 1, (0,255,0), 3)


    cv.imshow("motion detection", frame1)

    frame1= frame2
    ret, frame2 = cap.read()
    l=0

    if cv.waitKey(1)==27:
        break

cap.release()

cv.destroyAllWindows()

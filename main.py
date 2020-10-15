import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

ret, frame1 = cap.read()
ret, frame2 = cap.read()
l=0
while cap.isOpened():
    
    #difference between the first and second frame video
    
    difference = cv.absdiff(frame1, frame2)
    
    #then convert it into gray scale for better comparison of 0 and 1
    
    gray = cv.cvtColor(difference, cv.COLOR_BGR2GRAY)
    
    #After it we will Blur our picture to remove noise from it using gaussianblur method
    
    blurred = cv.GaussianBlur(gray, (5,5), 0)
    
    _, threshold1 = cv.threshold(blurred, 20, 255, cv.THRESH_BINARY)
    
    dilate_image = cv.dilate(threshold1, None, iterations=3)
    
    #After it we use contour function to find the contours of moving object
    
    contours, _ = cv.findContours(dilate_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    #for loop for counting the number of object in motion
    
    for contour1 in contours:
        if cv.contourArea(contour1) > 5000:
            l=l+1
    
    #for loop for displaying the contour that we have store using contour function
    
    for contour in contours:
        #Storing the  height width of the moving object
        
        (x, y, width, hieght) = cv.boundingRect(contour)
        
        
        if cv.contourArea(contour) < 5000:
            continue
            
        # Displaying rectangle around the motion object
        
        cv.rectangle(frame1, (x, y), (x+width, y+hieght), (0,255,255), 3)
        cv.putText(frame1, "Count: {}".format(str(l)), (10,20), cv.FONT_ITALIC, 1, (0,255,0), 3)


    cv.imshow("motion detection", frame1)
    
    #storing the second frame in first
    frame1= frame2
    # Storing the new frame in second frame
    
    ret, frame2 = cap.read()
    l=0

    if cv.waitKey(1)==27:
        break

cap.release()

cv.destroyAllWindows()

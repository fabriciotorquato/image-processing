# Standard imports
import cv2
import numpy as np
# Read image
img = cv2.imread("dados.jpg", cv2.IMREAD_COLOR)

gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#binary image
_,thresh = cv2.threshold(gray_image,247,247,cv2.THRESH_BINARY_INV)

#get contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(len(contours))
for contour in contours :
   
    area = cv2.contourArea(contour)
    if area >=1000:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.polylines(img, [box] ,  True,  (0, 123, 123),  3)
    
cv2.imshow("Blobs ",  img)
cv2.waitKey(0)
# Standard imports
import cv2
import numpy as np
# Read image
img = cv2.imread("dados.jpg", cv2.IMREAD_COLOR)

gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# binary image
_, thresh = cv2.threshold(gray_image, 247, 247, cv2.THRESH_BINARY_INV)

# get contours
contours, _ = cv2.findContours(
    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(len(contours))

# Set up the detector with default parameters.
detector = cv2.SimpleBlobDetector_create()
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.8
thickness = cv2.LINE_AA

for contour in contours:

    area = cv2.contourArea(contour)
    if area >= 1000:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.polylines(img, [box],  True,  (0, 123, 123),  3)
        
        x,y,w,h = cv2.boundingRect(contour)
        dado = img[y:y+h,x:x+w]

        # Detect blobs.
        keypoints = detector.detect(dado)

        # Get text siza
        text = str(len(keypoints))
        size = cv2.getTextSize(text, font, font_scale, thickness)
        text_width = size[0][0]
        text_height = size[0][1]
        cv2.putText(img,text,(int(x+(w/2)-(text_width/2)),int(y+(h/2)+(text_height/2))), font, font_scale,(0,0,255),2,thickness)

cv2.imshow("Blobs ",  img)
cv2.waitKey(0)

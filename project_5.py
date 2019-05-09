import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread('data/sudoku/sudoku1.jpg', cv2.IMREAD_GRAYSCALE)

tresholdMin = 30
tresholdMax = 60

kernelSize = (3, 3)

# filtro = cv2.GaussianBlur(img,kernelSize,0)
_, thresh = cv2.threshold(img, 247, 247, cv2.THRESH_BINARY_INV)
canny = cv2.erode(thresh, kernelSize, iterations=1)

# edges = cv2.erode(edges, kernelSize, iterations=1)

sobel = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=3)
sobel = cv2.erode(sobel, kernelSize, iterations=1)

# sobel = sobel - canny
# sobel = cv2.Sobel(sobel,cv2.CV_8U,0,1,ksize=3)


# get contours
contours, _ = cv2.findContours(
    sobel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

areas = []
for contour in contours:
    area = cv2.contourArea(contour)
    if area > 0 :
        areas.append(area)

areas = np.asarray(areas)
min = np.mean(areas)
print(np.mean(areas))

for contour in contours:

    area = cv2.contourArea(contour)
   
    if area >= 80 and area <= 300:
        print(area)
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.polylines(img, [box],  True,  (0, 123, 123),  3)


cv2.imshow('A',  canny)
cv2.imshow('B',  sobel)
cv2.imshow('C',  img)

cv2.waitKey()

cv2.destroyAllWindows()

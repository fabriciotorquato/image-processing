import cv2
import numpy as np

img = cv2.imread('data/sudoku/sudoku7.jpg', cv2.IMREAD_GRAYSCALE)

kernelSize = (5, 5)

_, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
thresh = cv2.erode(thresh, kernelSize, iterations=5)

rows, cols = thresh.shape
mask = np.zeros((rows+2, cols+2), np.uint8)
cv2.floodFill(thresh, mask, (0, 0),0)

im_floodfill_inv = cv2.bitwise_not(thresh)
rows, cols = im_floodfill_inv.shape
mask = np.zeros((rows+2, cols+2), np.uint8)
cv2.floodFill(im_floodfill_inv, mask, (0, 0),110)

_, thresh = cv2.threshold(im_floodfill_inv, 127, 255, cv2.THRESH_BINARY)

contours, _ = cv2.findContours(
    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


#   Na imagem 8 usar limite = 15
#   Nos outros usar 40
lim = 40

for contour in contours:

    area = cv2.contourArea(contour)

    if area >= lim:
        x,y,w,h = cv2.boundingRect(contour)
        cv2.rectangle(img,(x-5,y-5),(x+w+5,y+h+5),(127,127,128),2)

cv2.imshow('Z',  img)

cv2.waitKey()

cv2.destroyAllWindows()

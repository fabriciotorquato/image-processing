import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread('data/sudoku/sudoku1.jpg', cv2.IMREAD_GRAYSCALE)

tresholdMin = 100
tresholdMax = 200

kernelSize = (5,5)

# filtro = cv2.GaussianBlur(img,kernelSize,0)

# edges = cv2.Canny(filtro,tresholdMin,tresholdMax)

sobel = cv2.Sobel(img,cv2.CV_8U,1,1,ksize=5)

cv2.imshow('SobelX',  sobel)

cv2.imshow('Original', img)
# cv2.imshow('Canny',edges)

cv2.waitKey()

cv2.destroyAllWindows()
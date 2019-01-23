## Defined threshold ranges for HSV image to segment image (frog / background)

import numpy as np
import cv2
import matplotlib.pyplot as plt

# read and enhance image
img = cv2.imread('BspImSmall.JPG')
gaussian = cv2.GaussianBlur(img, (5, 5), 0)

# convert from color RGB to HSV
hsv = cv2.cvtColor(gaussian, cv2.COLOR_BGR2HSV)

# set ranges for threshold
lower_frog = np.array([0, 10, 0])
upper_frog = np.array([30, 255, 255])
# create binary image from threshold
output = cv2.inRange(hsv, lower_frog, upper_frog)

# show binary image
cv2.namedWindow('color_threshold', cv2.WINDOW_NORMAL)
cv2.imshow('color_threshold', output)
cv2.waitKey()

# enhance binary image with morphological Expressions
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

closing = cv2.morphologyEx(output, cv2.MORPH_CLOSE, kernel)
opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)


kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))

closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))

# opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)


kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11,11))

closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)


kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))

closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

mask = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel, iterations=1)

# show enhanced binary
cv2.namedWindow('morph', cv2.WINDOW_NORMAL)
cv2.imshow('morph', mask)
cv2.waitKey()

#
# contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(mask, contours, -1, (255,255,0), 3)
# cv2.namedWindow('contours', cv2.WINDOW_NORMAL)
# cv2.imshow('contours', mask)
# cv2.waitKey()



# res = cv2.bitwise_and(img, img, mask = mask)
#
# cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
# cv2.imshow('mask', res)
# cv2.waitKey()


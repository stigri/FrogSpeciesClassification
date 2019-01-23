## Attempt to remove background by using Canny Edge detector
## Too many contours found, how to find the right one?

import numpy as np
import cv2
from matplotlib import pyplot as plt

# read color image (1), read greyscale image (0)
image1 = cv2.imread('BspImSmall.JPG', 1)
image0 = cv2.imread('BspImSmall.JPG', 0)

# smooth image with gaussian blur
gaussian = cv2.GaussianBlur(image0, (5, 5), 0)

# morphological operators
kernel = np.ones((5,5),np.uint8)
opening = cv2.morphologyEx(gaussian, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

# Canny Edge operator
edges = cv2.Canny(closing,0,100)
plt.subplot(121), plt.imshow(image0, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(edges, cmap='gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()

# trying to find contours not working properly
(cnts, _) = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
frogCnt = cnts[0]

cv2.drawContours(image0, [frogCnt], -1, (0, 255, 0), 3)
cv2.imshow("Frog", image0)
cv2.waitKey(0)

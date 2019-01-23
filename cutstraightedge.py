
########################################################################################################################
# imports single image and reads exif data
# removes ruler by identifying the first pixel belonging to the ruler and cutting a straight edge
# removes parts of the frog if ruler and frog overlap
# uses gaussian filter and then creates binary from calculated threshold (Otsus method)
# enhances binary with morphological operators
# finds max connected component from binary and creates frogmask (with floodfill)
# uses frogmask on colorpicture
########################################################################################################################

import numpy as np
import cv2
import getexif
import pprint
from scipy import ndimage
# from matplotlib import pyplot as plt
# import histogrammatching as hismatch

# TODO: read through files with pictures and run code on them
# TODO: useful to initiate class for frog pictures including resized pictures, exif data, mask, masked picture?

# get exif data
exifData = getexif.get_exif('BspIm.JPG')
# prints exif data
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(exifData)

# read color image (1), read greyscale image (0) and convert color image to green image
image1 = cv2.imread('BspImRot.JPG', 1)
image0 = cv2.imread('BspIm.JPG', 0)

# calculate and plot histogram for all colorspaces of color image (0: blue, 1: green, 2: red) and for greyscale image
# color = ('b','g','r')
# for i, col in enumerate(color):
#     histogram = cv2.calcHist([image1],[i],None,[256],[0,256])
#     plt.plot(histogram, color = col)
#     plt.xlim([0,256])
# plt.show()

# plt.hist(image0.ravel(),256,[0,256]); plt.show()

# get shape of image
height, width = image0.shape

# if image is in landscape format
if height < width:
    # resize images to a width of 1500 pixel (resulting height: 1000 pixel)
    r = 1500.0 / width
    dim = (1500, int(height * r))
    image0 = cv2.resize(image0, dim, interpolation=cv2.INTER_AREA)
    image1 = cv2.resize(image1, dim, interpolation=cv2.INTER_AREA)

    # rotate to portrait format (ruler needs to be on the left side of the picture)
    image1 = ndimage.rotate(image1, -90)
    image0 = ndimage.rotate(image0, -90)

# if image is in portrait format
else:
    # resize to same dimensions as resized landscape format
    r = 1000.0 / width
    dim = (1000, int(height * r))
    image0 = cv2.resize(image0, dim, interpolation=cv2.INTER_AREA)
    image1 = cv2.resize(image1, dim, interpolation=cv2.INTER_AREA)

# if pixel on right upper corner is not white (ruler is on the right)
if image0[1:1,1:width-1] < 230:
    # keep format but rotate ruler to the left
    image1 = ndimage.rotate(image1, 180)
    image0 = ndimage.rotate(image0, 180)

# use green channel image for further calculations to see if it works better than greyscale image
# image0 = image1[:,:,1]

# show images
# cv2.namedWindow('frog color', cv2.WINDOW_NORMAL)
# cv2.imshow('frog color', image1)
# cv2.waitKey()
# cv2.namedWindow('frog grayscale', cv2.WINDOW_NORMAL)
# cv2.imshow('frog grayscale', image0)
# cv2.waitKey()

# destroy windows
cv2.destroyAllWindows()

# attempt to enhance picture using histogram matching
# template = cv2.imread('BspIm.JPG', 1)
# template = template[:,:,1]
# matched = hismatch.hist_match(image1[:,:,1], template)
# image0 = np.array(matched, dtype=np.uint8)

# smooth image with gaussian blur
gaussian = cv2.GaussianBlur(image0, (3, 3), 0)
# cv2.namedWindow('gaussian_image', cv2.WINDOW_NORMAL)
# cv2.imshow('gaussian_image', gaussian)
# cv2.waitKey()

# get number of columns and rows
cols, rows = image0.shape

# get last row of image
lastrow = image0[cols-1, :]

# reverse last row to iterate reversely over array
lastrow_rev = lastrow[::-1]

# iterate over last row (from right to left) and find index of first pixel which is not white (first pixel of ruler)
n = rows
for i in lastrow_rev:
    if i > 240:
        n = n-1
    else:
        break

# fill area of ruler white
gaussian[:, :n] = 255

# turn all pixels with value > 235 to white (255)
for x in np.nditer(gaussian, op_flags=['readwrite']):
    if x[...] > 235:
        x[...] = 255

# cv2.namedWindow('cropped_image', cv2.WINDOW_NORMAL)
# cv2.imshow('cropped_image', gaussian)
# cv2.waitKey()

# clahe = cv2.createCLAHE(clipLimit=90.0, tileGridSize=(8,8))
# gaussian = clahe.apply(gaussian)

# calculate threshold with Otsu's method to get binary image
# ret, thresh = cv2.threshold(gaussian, 0, 256, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# use adaptive threshold for threshold calculation to get binary image
thresh = cv2.adaptiveThreshold(gaussian, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4)

# morphological operators to remove artifacts inside frog area
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

kernel = np.ones((7, 7), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

# perform a connected components analysis on the binary
# initialize mask to store largest component
outputCC = cv2.connectedComponents(mask)
num_labels = outputCC[0]
# print (num_labels)

labels = outputCC[1]
frogmask = np.zeros(mask.shape, dtype='uint8')
maxnumPixel = 0

# loop over the unique components
for label in np.unique(labels):
    # ignore the background label
    if label == 0:
        continue

    # construct labelmask and count number of pixels
    labelsmask = np.zeros(mask.shape, dtype='uint8')
    labelsmask[labels == label] = 255
    numPixel = cv2.countNonZero(labelsmask)

    # find max connected component
    if numPixel > maxnumPixel:
        labelmask = np.zeros(mask.shape, dtype='uint8')
        labelmask[labels == label] = 255
        maxnumPixel = numPixel

# add max connected component to frogmask
frogmask = cv2.add(frogmask, labelmask)

# different approach to get mask via finding and drawing max contour of frogmask
# TODO: fill found contour
# _, contours, hierarchy = cv2.findContours(frogmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# maxcontour = max(contours, key = cv2.contourArea)
# mask = np.zeros(frogmask.shape, dtype='uint8')
# cv2.drawContours(mask, maxcontour, -1, (255, 0, 0), 4)

# see https://www.learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/ for the following approach
# copy the thresholded image to use flood filling on frogmask
frogmask_floodfill = frogmask.copy()

# mask used to flood filling.
# notice the size needs to be 2 pixels larger than the original image.
h, w = frogmask.shape[:2]
mask_floodfilling = np.zeros((h + 2, w + 2), np.uint8)

# floodfill from point (0, 0)
cv2.floodFill(frogmask_floodfill, mask_floodfilling, (0, 0), 255)

# invert floodfilled image
frogmask_floodfill_inv = cv2.bitwise_not(frogmask_floodfill)

# combine the two images to get the foreground
frogmask_out = frogmask | frogmask_floodfill_inv

# isplay images
cv2.namedWindow('Thresholded Image', cv2. WINDOW_NORMAL)
# cv2.namedWindow('Floodfilled Image', cv2. WINDOW_NORMAL)
# cv2.namedWindow('Inverted Floodfilled Image', cv2. WINDOW_NORMAL)
cv2.namedWindow('Foreground', cv2. WINDOW_NORMAL)
cv2.imshow("Thresholded Image", frogmask)
# cv2.imshow("Floodfilled Image", frogmask_floodfill)
# cv2.imshow("Inverted Floodfilled Image", frogmask_floodfill_inv)
cv2.imshow("Foreground", frogmask_out)
cv2.waitKey()
cv2.destroyAllWindows()

# use mask on color image
masked_image = cv2.bitwise_and(image1, image1, mask=frogmask_out)

cv2.namedWindow('masked image', cv2. WINDOW_NORMAL)
cv2.imshow('masked image', masked_image)
cv2.waitKey()
cv2.destroyAllWindows()

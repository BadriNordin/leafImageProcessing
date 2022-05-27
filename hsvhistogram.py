from operator import xor
from unittest import result
import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
from pyparsing import And
import os

orileaf = cv2.imread("C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Documents/UTP/4th2nd/FYP/Site/visit 1/idk/IMG_1491.JPG")
# orileaf = cv2.cvtColor(orileaf, cv2.COLOR_BGR2RGB)

testh = cv2.imread("C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Documents/UTP/4th2nd/FYP/Site/visit 1/idk/testhist.JPG")

testl = cv2.imread("C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Documents/UTP/4th2nd/FYP/Site/visit 1/idk/testleaf.JPG")

testg = cv2.imread("C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Documents/UTP/4th2nd/FYP/Site/visit 1/idk/testg.JPG")

hleaf = cv2.imread("C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Documents/UTP/4th2nd/FYP\Site/visit 1/healthy/IMG_1626.JPG", cv2.IMREAD_UNCHANGED)

orileaf2 = cv2.imread("C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Documents/UTP/4th2nd/FYP/Site/visit 1/idk/IMG_1502.JPG")

pilfcm = cv2.imread("C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Documents/UTP/4th2nd/FYP/Site/visit 1/idk/pilfcm.JPG")

cv2fcm = cv2.imread("C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Documents/UTP/4th2nd/FYP/Site/visit 1/idk/cv2fcm.JPG")

testsavecv2 = cv2.imread("C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Pictures/testsavecv2.JPG")

label10 = cv2.imread("C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Documents/UTP/4th2nd/FYP/Site/visit 1/testimages/testlabel10.jpg")

directory = r'C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Documents/UTP/4th2nd/FYP/Site/visit 1/testimages'
os.chdir(directory)

dim = (700,700)
# orileaf = cv2.resize(orileaf, dim, interpolation = cv2.INTER_AREA)
r,g,b =  cv2.split(orileaf)

# cv2fcm = cv2.resize(cv2fcm, dim, interpolation = cv2.INTER_AREA)
# pilfcm = cv2.resize(pilfcm, dim, interpolation = cv2.INTER_AREA)

# cvthsv = cv2.cvtColor(orileaf, cv2.COLOR_RGB2HSV)
cvthsv = cv2.cvtColor(orileaf, cv2.COLOR_RGB2HSV)
h,s,v = cv2.split(cvthsv)
hsv_split = np.concatenate((h,s,v), axis = 1)

maskh0 = cv2.inRange(h,85,120)
maskh = cv2.inRange(h,85,105.6)
maskh2 = cv2.inRange(h,85,95)
maskv1 = cv2.inRange(v,0,50)
maskv2 = cv2.inRange(v,76,255)
maskv3 = cv2.inRange(v,50,76)

maskt = maskh + maskv3
maskvt = maskh ^ maskv2 #XOR
maskvt2 = maskh & maskv2 #AND *
maskvt3 = maskh | maskv2 #OR

dis1 = cv2.inRange(h,97.6,98.6)
dis2 = cv2.inRange(h,99.6,105.6)
dis = dis1 + dis2

uleaf = cv2.inRange(h,85.65,102.6)
stem = cv2.inRange(h,85.66,95.62)
stem = cv2.bitwise_not(stem)
backgrass1 = cv2.inRange(s,10.94,12.94)
backgrass2 = cv2.inRange(s,13.9,96.7)
backgrass = backgrass1 + backgrass2
backgrass = cv2.bitwise_not(backgrass)
moss = cv2.inRange(h,75.6,89.6)
moss = cv2.bitwise_not(moss)

maskh3 = maskvt2 & backgrass & stem & moss

# res = cv2.bitwise_and(orileaf,orileaf,mask= maskh0)
# res2 = cv2.bitwise_and(orileaf,orileaf,mask= maskvt2)
# res3 = cv2.bitwise_and(orileaf,orileaf,mask= maskh3)

#label masks

l1low = (119, 145, 149) 
l1high = (120, 146, 150)
l1 = cv2.inRange(label10, l1low, l1high)

l3low = (1, 1, 2) 
l3high = (1.57, 1.91, 2.55)
l3 = cv2.inRange(label10, l3low, l3high)

l10low = (91, 119, 123) 
l10high = (93, 121, 125)
l10 = cv2.inRange(label10, l10low, l10high)

checklabel = cv2.bitwise_and(label10, label10, mask=l3)

cv2.imshow('res',checklabel)
# cv2.imshow('res2',res2)
# cv2.imshow('res3',res3)

# plt.subplot(1,2,1)
# plt.imshow(l1low)

# plt.subplot(1,2,2)
# plt.imshow(res2)

# cv2.imwrite('testsaveres1.jpg', res)

# maksh3 for disease only mask
# maskvt2 to see whole leaf withits shape but background still present

#########################################################################
# #k-means
# kimage = cv2.cvtColor(res3, cv2.COLOR_BGR2RGB)
# temp = cv2.cvtColor(orileaf, cv2.COLOR_BGR2RGB)
# plt.subplot(1,3,1)
# plt.imshow(temp)

# # reshape the image to a 2D array of pixels and 3 color values (RGB)
# pixel_values = kimage.reshape((-1, 3))
# # convert to float
# pixel_values = np.float32(pixel_values)
# print(pixel_values.shape)

# # define stopping criteria
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

# # number of clusters (K)
# k = 3
# _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# # convert back to 8 bit values
# centers = np.uint8(centers)

# # flatten the labels array
# labels = labels.flatten()

# # convert all pixels to the color of the centroids
# segmented_image = centers[labels.flatten()]

# # reshape back to the original image dimension
# segmented_image = segmented_image.reshape(kimage.shape)

# # show the image
# plt.subplot(1,3,2)
# plt.imshow(segmented_image)

# # disable only the cluster number 2 (turn the pixel into black)
# masked_image = np.copy(kimage)

# # convert to the shape of a vector of pixel values
# masked_image = masked_image.reshape((-1, 3))
# # color (i.e cluster) to disable
# cluster = 0 #black
# cluster2 = 2 # light brown
# cluster3 = 1 # dark brown/disease
# masked_image[labels == cluster] = [0, 0, 0]
# # masked_image[labels == cluster2] = [0, 0, 0]
# masked_image[labels == cluster3] = [0, 0, 0]

# # convert back to original shape
# masked_image = masked_image.reshape(kimage.shape)

# # show the image
# plt.subplot(1,3,3)
# plt.imshow(masked_image)

plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows
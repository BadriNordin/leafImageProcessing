from unittest import result
import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors

orileaf = cv2.imread("C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Documents/UTP/4th2nd/FYP/Site/visit 1/idk/IMG_1491.JPG")
# orileaf = cv2.cvtColor(orileaf, cv2.COLOR_BGR2RGB)

testh = cv2.imread("C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Documents/UTP/4th2nd/FYP/Site/visit 1/idk/testhist.JPG")

testl = cv2.imread("C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Documents/UTP/4th2nd/FYP/Site/visit 1/idk/testleaf.JPG")

testg = cv2.imread("C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Documents/UTP/4th2nd/FYP/Site/visit 1/idk/testg.JPG")

hleaf = cv2.imread("C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Documents/UTP/4th2nd/FYP\Site/visit 1/healthy/IMG_1626.JPG", cv2.IMREAD_UNCHANGED)

orileaf2 = cv2.imread("C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Documents/UTP/4th2nd/FYP/Site/visit 1/idk/IMG_1502.JPG")

dim = (1000,1000)
orileaf = cv2.resize(orileaf2, dim, interpolation = cv2.INTER_AREA)
r,g,b =  cv2.split(orileaf)

# cvthsv = cv2.cvtColor(orileaf, cv2.COLOR_RGB2HSV)
cvthsv = cv2.cvtColor(orileaf, cv2.COLOR_RGB2HSV)
h,s,v = cv2.split(cvthsv)
hsv_split = np.concatenate((h,s,v), axis = 1)

maskh = cv2.inRange(h,85,120)
maskh2 = cv2.inRange(h,85,95)
maskv = cv2.inRange(v,120,130)

maskt = maskh + maskh2
res = cv2.bitwise_and(orileaf,orileaf,mask= maskh)
res2 = cv2.bitwise_and(orileaf,orileaf,mask= maskv)
res3 = cv2.bitwise_and(orileaf,orileaf,mask= maskt)

# cv2.imshow('h',h)
# cv2.imshow('mask',maskh)
# cv2.imshow('result',res)

# plt.hist(res.ravel(),256,[0,255])

# idk = cv2.bitwise_and(orileaf,orileaf, mask= maskh)
cv2.imshow('maskh',res)
cv2.imshow('maskv',res2)
cv2.imshow('maskt',res3)

# crop image to get bottom part, apply filter and find range to segment out
# plot and cv2 view images differently, focus o cv2


plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows

# 36-68
# 64-105
# 125-167

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

orileaf = cv2.imread("C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Documents/UTP/4th2nd/FYP/Site/visit 1/idk/IMG_1491.JPG")
# orileaf = cv2.cvtColor(orileaf, cv2.COLOR_BGR2RGB)

testh = cv2.imread("C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Documents/UTP/4th2nd/FYP/Site/visit 1/idk/testhist.JPG")

testl = cv2.imread("C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Documents/UTP/4th2nd/FYP/Site/visit 1/idk/testleaf.JPG")

testg = cv2.imread("C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Documents/UTP/4th2nd/FYP/Site/visit 1/idk/testg.JPG")

hleaf = cv2.imread("C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Documents/UTP/4th2nd/FYP\Site/visit 1/healthy/IMG_1626.JPG", cv2.IMREAD_UNCHANGED)

orileaf2 = cv2.imread("C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Documents/UTP/4th2nd/FYP/Site/visit 1/idk/IMG_1502.JPG")

dim = (1500,1300)
orileaf = cv2.resize(orileaf, dim, interpolation = cv2.INTER_AREA)
r,g,b =  cv2.split(orileaf)

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

res = cv2.bitwise_and(orileaf,orileaf,mask= maskvt2)
res2 = cv2.bitwise_and(orileaf,orileaf,mask= dis)
res3 = cv2.bitwise_and(orileaf,orileaf,mask= maskh3)

# cv2.imshow('res',res)
# cv2.imshow('res2',res2)
cv2.imshow('res3',res3)

# maksh3 for disease only mask
# maskvt2 to see whole leaf withits shape but background still present


plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows

# 36-68
# 64-105
# 125-167

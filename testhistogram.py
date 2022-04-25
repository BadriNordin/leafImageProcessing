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

testl2 = cv2.imread("C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Documents/UTP/4th2nd/FYP/Site/visit 1/idk/testleaf2.JPG")

testg = cv2.imread("C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Documents/UTP/4th2nd/FYP/Site/visit 1/idk/testg.JPG")

moss = cv2.imread("C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Documents/UTP/4th2nd/FYP/Site/visit 1/idk/IMG_1502 moss.JPG")

disease = cv2.imread("C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Documents/UTP/4th2nd/FYP/Site/visit 1/idk/disease.JPG")

g1 = cv2.imread("C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Documents/UTP/4th2nd/FYP/Site/visit 1/idk/ground1.JPG")

backgrass = cv2.imread("C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Documents/UTP/4th2nd/FYP/Site/visit 1/idk/backgrass.JPG")

dleaf = cv2.imread("C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Documents/UTP/4th2nd/FYP/Site/visit 1/idk/dleaf.JPG")

stem = cv2.imread("C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Documents/UTP/4th2nd/FYP/Site/visit 1/idk/stem.JPG")

dim = (700,700)
orileaf = cv2.resize(disease, dim, interpolation = cv2.INTER_AREA)
r,g,b =  cv2.split(orileaf)

cvthsv = cv2.cvtColor(moss, cv2.COLOR_RGB2HSV)
h,s,v = cv2.split(cvthsv)
hsv_split = np.concatenate((h,s,v), axis = 1)

# #3D plot
# fig = plt.figure()
# axis = fig.add_subplot(1,1,1, projection = "3d")

# pixel_colors = orileaf.reshape((np.shape(orileaf)[0]*np.shape(orileaf)[1],3))
# norm = colors.Normalize(vmin=-1.,vmax=1.)
# norm.autoscale(pixel_colors)
# pixel_colors = norm(pixel_colors).tolist()

# axis.scatter(r.flatten(),g.flatten(),b.flatten(), facecolors=pixel_colors, marker=".")
# axis.set_xlabel("Red")
# axis.set_ylabel("Green")
# axis.set_zlabel("Blue")

# cv2.imshow("B",b)
# cv2.imshow("G",g)
# cv2.imshow("R",r)
# cv2.imshow("H",h)
# cv2.imshow("S",s)
# cv2.imshow("V",v)

# 1st plot, the image
plt.subplot(1,4,1)
# plt.subplot(1,3,1)
plt.imshow(cvthsv)

# #2nd plot R value
# plt.subplot(2,2,2)
# plt.hist(r.ravel(),256,[0,255])
# plt.xlabel("R")

# #3rd plot G value
# plt.subplot(2,2,3)
# plt.hist(g.ravel(),256,[0,255])
# plt.xlabel("G")

# #4th plot b value
# plt.subplot(2,2,4)
# plt.hist(b.ravel(),256,[0,255])
# plt.xlabel("B")

#2nd plot H value
plt.subplot(1,4,2)
plt.hist(h.ravel(),256,[0,255])
plt.xlabel("H")

#3rd plot S value
plt.subplot(1,4,3)
plt.hist(s.ravel(),256,[0,255])
plt.xlabel("S")

#4th plot V value
plt.subplot(1,4,4)
plt.hist(v.ravel(),256,[0,255])
plt.xlabel("V")

# #shows the background
# low = (50,85,30)
# high = (140,200,127)
# mask = cv2.inRange(orileaf, low, high)
# res = cv2.bitwise_and(orileaf, orileaf, mask=mask)
# plt.subplot(1,3,2)
# plt.imshow(res)

# #show foreground
# lowa = (0,0,0)
# higha = (49,84,29)
# lowb = (141,201,128)
# highb = (250,250,250)

# maska = cv2.inRange(orileaf,lowa,higha)
# maskb = cv2.inRange(orileaf,lowb,highb)
# maskfinal = maska + maskb
# res2 = cv2.bitwise_and(orileaf, orileaf, mask=maskfinal)
# plt.subplot(1,3,3)
# plt.imshow(res2)

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows

#50-90
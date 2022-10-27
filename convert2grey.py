import cv2
import numpy as np
import os

directory = r'C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Documents/UTP/4th2nd/FYP/Site/visit 1/grayscale'
os.chdir(directory)

healthy1 = cv2.imread("C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Documents/UTP/4th2nd/FYP/Site/visit 1/healthy/IMG_1607.JPG")

healthy2 = cv2.imread("C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Documents/UTP/4th2nd/FYP/Site/visit 1/healthy/IMG_1610.JPG")

healthy3 = cv2.imread("C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Documents/UTP/4th2nd/FYP/Site/visit 1/healthy/IMG_1621.JPG")

healthy4 = cv2.imread("C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Documents/UTP/4th2nd/FYP/Site/visit 1/healthy/IMG_1694.JPG")

healthy5 = cv2.imread("C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Documents/UTP/4th2nd/FYP/Site/visit 1/healthy/IMG_1769.JPG")

disease1 = cv2.imread("C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Documents/UTP/4th2nd/FYP/Site/visit 1/unhealthy/IMG_1491.JPG")

disease2 = cv2.imread("C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Documents/UTP/4th2nd/FYP/Site/visit 1/unhealthy/IMG_1592.JPG")

disease3 = cv2.imread("C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Documents/UTP/4th2nd/FYP/Site/visit 1/unhealthy/IMG_1596.JPG")

disease4 = cv2.imread("C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Documents/UTP/4th2nd/FYP/Site/visit 1/unhealthy/IMG_1600.JPG")

disease5 = cv2.imread("C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Documents/UTP/4th2nd/FYP/Site/visit 1/unhealthy/IMG_1498.JPG")

image = disease5

image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
h,s,v = cv2.split(image)

maskh0 = cv2.inRange(h,85,120)

image = cv2.bitwise_and(image,image, mask=maskh0)

# # get (i, j) positions of all RGB pixels that are black (i.e. [0, 0, 0])
# black_pixels = np.where(
#     (image[:, :, 0] == 0) & 
#     (image[:, :, 1] == 0) & 
#     (image[:, :, 2] == 0)
# )

# # set those pixels to white
# image[black_pixels] = [255, 255, 255]

grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow('',grey)
cv2.imwrite('disease5b.jpg', grey)

cv2.waitKey(0)
cv2.destroyAllWindows()
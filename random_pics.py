import cv2
import matplotlib.pyplot as plt

image = cv2.imread("C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Documents/UTP/4th2nd/FYP/svm/dataset3/Unhealthy/IMG_1491.jpg")
imaget = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
imaget2 = cv2.resize(imaget,(700,700))

image2 = cv2.resize(image,(700,700))
image3 = cv2.cvtColor(image2, cv2.COLOR_RGB2HSV)
# image3 = cv2.resize(image3,(700,700))

# h,s,v = cv2.split(image3)
# maskh0 = cv2.inRange(h,85,120)

# leaf = cv2.imread("C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Documents/UTP/4th2nd/FYP/Site/visit 1/testimages/testleaf.jpg")
# leafty = cv2.imread("C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Documents/UTP/4th2nd/FYP/Site/visit 1/testimages/dleaf.jpg")
# leaftt = cv2.cvtColor(leafty, cv2.COLOR_BGR2RGB)
# leaf = leafty
# # leaf = cv2.cvtColor(leaf, cv2.COLOR_BGR2RGB)
# leaf2 = cv2.resize(leaf,(700,700))
# leaf3 = cv2.cvtColor(leaf2, cv2.COLOR_RGB2HSV)
# hl,sl,vl = cv2.split(leaf3)

# healthy = cv2.imread("C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Documents/UTP/4th2nd/FYP/svm/dataset3/Healthy/IMG_1762.JPG")
# healthy = cv2.resize(healthy,(700,700))
# healthy = cv2.cvtColor(healthy, cv2.COLOR_BGR2RGB)
# h2,s2,v2 = cv2.split(healthy)
# maskh0 = cv2.inRange(h2,85,120)

# plt.subplot(1,2,1)
# plt.imshow(healthy)
# plt.title('Original Image')

# masked = cv2.bitwise_and(healthy,healthy, mask=maskh0) 
# plt.subplot(1,2,2)
# plt.imshow(masked)
# plt.title('Masked Image')

# resize and convert to HSV plots
plt.subplot(1,3,1)
plt.imshow(imaget)
plt.title('Original Image')

plt.subplot(1,3,2)
plt.imshow(imaget2)
plt.title('700x700 Image')

plt.subplot(1,3,3)
plt.imshow(image3)
plt.title('HSV Image')

# plt.subplot(2,4,1)
# plt.imshow(imaget)
# plt.title('Original Image')

# plt.subplot(2,4,2)
# plt.hist(h.ravel(),256,[0,255])
# plt.title('H Component')

# plt.subplot(2,4,3)
# plt.hist(s.ravel(),256,[0,255])
# plt.title('S Component')

# plt.subplot(2,4,4)
# plt.hist(v.ravel(),256,[0,255])
# plt.title('V Component')

# plt.subplot(2,4,5)
# plt.imshow(leaftt)
# plt.title('Leaf')

# plt.subplot(2,4,6)
# plt.hist(hl.ravel(),256,[0,255])
# plt.title('H Component')

# plt.subplot(2,4,7)
# plt.hist(sl.ravel(),256,[0,255])
# plt.title('S Component')

# plt.subplot(2,4,8)
# plt.hist(vl.ravel(),256,[0,255])
# plt.title('V Component')

plt.tight_layout()
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows
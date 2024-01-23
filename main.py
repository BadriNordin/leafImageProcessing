import cv2
import numpy as np
import matplotlib as plt

# orileaf = cv2.imread("C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Documents/UTP/4th2nd/FYP/Site/visit 1/idk/IMG_1491.JPG", cv2.IMREAD_UNCHANGED)
# orileaf = cv2.cvtColor(orileaf, cv2.COLOR_BGR2RGB)
orileaf = cv2.imread("C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Documents/UTP/4th2nd/FYP/svm/dataset3/Unhealthy/IMG_1491.jpg")


hleaf = cv2.imread("C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Documents/UTP/4th2nd/FYP\Site/visit 1/healthy/IMG_1626.JPG", cv2.IMREAD_UNCHANGED)
hleaf = cv2.cvtColor(hleaf, cv2.COLOR_BGR2RGB)

print("Original Dimensions: ",orileaf.shape)

dim = (700,700)
resized = cv2.resize(orileaf, dim, interpolation = cv2.INTER_AREA)
hresized = cv2.resize(hleaf, dim, interpolation = cv2.INTER_AREA)
print("Resized Dimensions: ",resized.shape)

cvthsv = cv2.cvtColor(resized, cv2.COLOR_RGB2HSV)
h,s,v = cv2.split(cvthsv)
hsv_split = np.concatenate((h,s,v), axis = 1)

hcvthsv = cv2.cvtColor(hresized, cv2.COLOR_RGB2HSV)
hh,hs,hv = cv2.split(hcvthsv)
hhsv_split = np.concatenate((hh,hs,hv), axis = 1)

cvthsl = cv2.cvtColor(resized, cv2.COLOR_RGB2HLS)
h2,l,s2 = cv2.split(cvthsl)
hsl_split = np.concatenate((h2,s2,l), axis = 1)

# cv2.imshow("Resized Image: ",resized)
# cv2.imshow("Healthy Resized Image: ",hresized)
# cv2.imshow("HSV Image: ",cvthsv)
# cv2.imshow("HSL Image: ",cvthsi)
# cv2.imshow("HSV Image: ",hsv_split)
# cv2.imshow("Healthy HSV Image: ",hhsv_split)
#cv2.imshow("HSL Image: ",hsl_split)
cv2.imshow("H Component:",h)
cv2.imshow("S Component:",s)
cv2.imshow("V Component:",v)
# cv2.imshow("H Component:",h2)
# cv2.imshow("S Component:",s2)
# cv2.imshow("L Component:",l)
cv2.waitKey(0)
cv2.destroyAllWindows
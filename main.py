import cv2

orileaf = cv2.imread("C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Documents/UTP/4th2nd/FYP/Site/visit 1/idk/IMG_1491.JPG", cv2.IMREAD_UNCHANGED)
print("Original Dimensions: ",orileaf.shape)

dim = (700,700)
resized = cv2.resize(orileaf, dim, interpolation = cv2.INTER_AREA)
print("Resized Dimensions: ",resized.shape)

cvthsv = cv2.cvtColor(resized, cv2.COLOR_RGB2HSV)

cv2.imshow("Resized Image: ",resized)
cv2.imshow("HSV Image: ",cvthsv)
cv2.waitKey(0)
cv2.destroyAllWindows
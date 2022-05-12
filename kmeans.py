import cv2
import numpy as np
import matplotlib.pyplot as plt

orileaf = cv2.imread("C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Documents/UTP/4th2nd/FYP/Site/visit 1/idk/IMG_1491.JPG", cv2.IMREAD_UNCHANGED)
# orileaf = cv2.cvtColor(orileaf, cv2.COl)

pixel_values = orileaf.reshape((-1,3))
# pixel_values = np.float32(pixel_values)
print("rehsape: ",pixel_values.shape)
print("ori: ",orileaf.shape)
cv2.imshow("reshape",pixel_values)
cv2.imshow("ori",orileaf)
cv2.waitKey(0)
cv2.destroyAllWindows

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

k = 3
_, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# convert back to 8 bit values
centers = np.uint8(centers)

# flatten the labels array
labels = labels.flatten()

# convert all pixels to the color of the centroids
segmented_image = centers[labels.flatten()]

# # reshape back to the original image dimension
# segmented_image = segmented_image.reshape(orileaf.shape)
# # show the image
# plt.imshow(segmented_image)
# plt.show()

# disable only the cluster number 2 (turn the pixel into black)
masked_image = np.copy(orileaf)
# convert to the shape of a vector of pixel values
masked_image = masked_image.reshape((-1, 3))
# color (i.e cluster) to disable
cluster = 2
masked_image[labels == cluster] = [0, 0, 0]
# convert back to original shape
masked_image = masked_image.reshape(orileaf.shape)
# show the image
plt.imshow(masked_image)
plt.show()
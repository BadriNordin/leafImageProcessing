from msilib.schema import Directory
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
from fcmeans import FCM
from PIL import Image

directory = r'C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Documents/UTP/4th2nd/FYP/histograms'
os.chdir(directory)

np.set_printoptions(threshold=np.inf)

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

image = disease3
image = cv2.resize(image,(700,700))
image2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.subplot(1,3,1)
plt.imshow(image2)
plt.title('Original Image')

image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
h,s,v = cv2.split(image)
# plt.subplot(3,2,2)
# plt.imshow(image)

maskh0 = cv2.inRange(h,85,120)

#get what masked looks like
masked = cv2.bitwise_and(image,image, mask=maskh0) 
plt.subplot(1,3,2)
plt.imshow(masked)
plt.title('Masked')

#save maksed image

#get HSV graph
# plt.subplot(3,2,4)
# plt.hist(h.ravel(),256,[0,255])

#save HSV graph

#get what clustered looks like
#Transforming image into a data set
X = (
    np.asarray(masked)                              # convert a PIL image to np array
    .reshape((700*700, 3))                             # reshape the image to convert each pixel to an instance of a data set
)

#Creating and fitting the model
fcm = FCM(n_clusters=4)                           # create a FCM instance with 10 clusters
fcm.fit(X)
labeld_X = fcm.predict(X)                          # get the label of each data point

transformed_X = fcm.centers[labeld_X]              # pixel quantization into the centers
# print('\nfcm.centers:\n',fcm.centers)

quatized_array = (                                 #Converting and saving image
    transformed_X
    .astype('uint8')                               # convert data points into 8-bit unsigned integers
    .reshape((700, 700, 3))                            # reshape image
)

quatized_image = Image.fromarray(np.asarray(quatized_array))   # convert array into a PIL image object
# quatized_image.save('C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Documents/UTP/4th2nd/FYP/histograms/tempimg.jpg') # save image

# quatized_image.show()
plt.subplot(1,3,3)
plt.imshow(quatized_image)
plt.title('Clustered')

# image2 = np.array(Image.open('C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Documents/UTP/4th2nd/FYP/histograms/tempimg.jpg'))
# h2,s2,v2 = cv2.split(image2)

# image2 = Image.Image.split(quatized_image)
# h2 = image2[0]
# h2.show()

#save clustered image

#get clustered HSV graph
# plt.subplot(3,2,6)
# plt.hist(np.ravel(h2),256,[0,255])

# save clustered HSV graph

# combine to make subplots that make sense

# save the summary subplots

# cv2.imshow('',quatized_image)
# cv2.imwrite('disease5b.jpg', quatized_image)

plt.tight_layout()
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
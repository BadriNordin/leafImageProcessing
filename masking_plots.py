from msilib.schema import Directory
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
from fcmeans import FCM
from PIL import Image

disease1 = cv2.imread("C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Documents/UTP/4th2nd/FYP/Site/visit 1/unhealthy/IMG_1491.JPG")

disease2 = cv2.imread("C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Documents/UTP/4th2nd/FYP/Site/visit 1/unhealthy/IMG_1592.JPG")

disease3 = cv2.imread("C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Documents/UTP/4th2nd/FYP/Site/visit 1/unhealthy/IMG_1596.JPG")

disease4 = cv2.imread("C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Documents/UTP/4th2nd/FYP/Site/visit 1/unhealthy/IMG_1600.JPG")

disease5 = cv2.imread("C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Documents/UTP/4th2nd/FYP/Site/visit 1/unhealthy/IMG_1498.JPG")

image = disease1
image = cv2.resize(image,(700,700))
image2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.subplot(3,3,1)
plt.imshow(image2)
plt.title('Original Image')

image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
h,s,v = cv2.split(image)
# plt.subplot(3,2,2)
# plt.imshow(image)

maskh0 = cv2.inRange(h,85,120)

#get what masked looks like
masked = cv2.bitwise_and(image,image, mask=maskh0) 
plt.subplot(3,3,2)
plt.imshow(masked)
plt.xticks([100,200,300,400,500,600])
plt.yticks([100,200,300,400,500,600])
plt.title('Masked')

#get what clustered looks like
#Transforming image into a data set
X = (
    np.asarray(masked)                              # convert a PIL image to np array
    .reshape((700*700, 3))                             # reshape the image to convert each pixel to an instance of a data set
)

#Creating and fitting the model
fcm = FCM(n_clusters=12)                           # create a FCM instance with 10 clusters
fcm.fit(X)
labeld_X = fcm.predict(X)                          # get the label of each data point

transformed_X = fcm.centers[labeld_X]              # pixel quantization into the centers

quatized_array = (                                 #Converting and saving image
    transformed_X
    .astype('uint8')                               # convert data points into 8-bit unsigned integers
    .reshape((700, 700, 3))                            # reshape image
)

quatized_image = Image.fromarray(np.asarray(quatized_array))   # convert array into a PIL image object

plt.subplot(3,3,3)
plt.imshow(quatized_image)
plt.title('Clustered')

#######################################################################################################

image = disease2
image = cv2.resize(image,(700,700))
image2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.subplot(3,3,4)
plt.imshow(image2)

image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
h,s,v = cv2.split(image)
# plt.subplot(3,2,2)
# plt.imshow(image)

maskh0 = cv2.inRange(h,85,120)

#get what masked looks like
masked = cv2.bitwise_and(image,image, mask=maskh0) 
plt.subplot(3,3,5)
plt.imshow(masked)

#get what clustered looks like
#Transforming image into a data set
X = (
    np.asarray(masked)                              # convert a PIL image to np array
    .reshape((700*700, 3))                             # reshape the image to convert each pixel to an instance of a data set
)

#Creating and fitting the model
fcm = FCM(n_clusters=12)                           # create a FCM instance with 10 clusters
fcm.fit(X)
labeld_X = fcm.predict(X)                          # get the label of each data point

transformed_X = fcm.centers[labeld_X]              # pixel quantization into the centers

quatized_array = (                                 #Converting and saving image
    transformed_X
    .astype('uint8')                               # convert data points into 8-bit unsigned integers
    .reshape((700, 700, 3))                            # reshape image
)

quatized_image = Image.fromarray(np.asarray(quatized_array))   # convert array into a PIL image object

plt.subplot(3,3,6)
plt.imshow(quatized_image)

#################################################################################################

image = disease3
image = cv2.resize(image,(700,700))
image2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.subplot(3,3,7)
plt.imshow(image2)

image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
h,s,v = cv2.split(image)
# plt.subplot(3,2,2)
# plt.imshow(image)

maskh0 = cv2.inRange(h,85,120)

#get what masked looks like
masked = cv2.bitwise_and(image,image, mask=maskh0) 
plt.subplot(3,3,8)
plt.imshow(masked)

#get what clustered looks like
#Transforming image into a data set
X = (
    np.asarray(masked)                              # convert a PIL image to np array
    .reshape((700*700, 3))                             # reshape the image to convert each pixel to an instance of a data set
)

#Creating and fitting the model
fcm = FCM(n_clusters=12)                           # create a FCM instance with 10 clusters
fcm.fit(X)
labeld_X = fcm.predict(X)                          # get the label of each data point

transformed_X = fcm.centers[labeld_X]              # pixel quantization into the centers

quatized_array = (                                 #Converting and saving image
    transformed_X
    .astype('uint8')                               # convert data points into 8-bit unsigned integers
    .reshape((700, 700, 3))                            # reshape image
)

quatized_image = Image.fromarray(np.asarray(quatized_array))   # convert array into a PIL image object

plt.subplot(3,3,9)
plt.imshow(quatized_image)

##########################################################################################

# image = disease4
# image = cv2.resize(image,(700,700))
# image2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# plt.subplot(5,3,10)
# plt.imshow(image2)

# image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
# h,s,v = cv2.split(image)
# # plt.subplot(3,2,2)
# # plt.imshow(image)

# maskh0 = cv2.inRange(h,85,120)

# #get what masked looks like
# masked = cv2.bitwise_and(image,image, mask=maskh0) 
# plt.subplot(5,3,11)
# plt.imshow(masked)

# #get what clustered looks like
# #Transforming image into a data set
# X = (
#     np.asarray(masked)                              # convert a PIL image to np array
#     .reshape((700*700, 3))                             # reshape the image to convert each pixel to an instance of a data set
# )

# #Creating and fitting the model
# fcm = FCM(n_clusters=4)                           # create a FCM instance with 10 clusters
# fcm.fit(X)
# labeld_X = fcm.predict(X)                          # get the label of each data point

# transformed_X = fcm.centers[labeld_X]              # pixel quantization into the centers

# quatized_array = (                                 #Converting and saving image
#     transformed_X
#     .astype('uint8')                               # convert data points into 8-bit unsigned integers
#     .reshape((700, 700, 3))                            # reshape image
# )

# quatized_image = Image.fromarray(np.asarray(quatized_array))   # convert array into a PIL image object

# plt.subplot(5,3,12)
# plt.imshow(quatized_image)

# ###############################################################################

# image = disease5
# image = cv2.resize(image,(700,700))
# image2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# plt.subplot(5,3,13)
# plt.imshow(image2)

# image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
# h,s,v = cv2.split(image)
# # plt.subplot(3,2,2)
# # plt.imshow(image)

# maskh0 = cv2.inRange(h,85,120)

# #get what masked looks like
# masked = cv2.bitwise_and(image,image, mask=maskh0) 
# plt.subplot(5,3,14)
# plt.imshow(masked)

# #get what clustered looks like
# #Transforming image into a data set
# X = (
#     np.asarray(masked)                              # convert a PIL image to np array
#     .reshape((700*700, 3))                             # reshape the image to convert each pixel to an instance of a data set
# )

# #Creating and fitting the model
# fcm = FCM(n_clusters=4)                           # create a FCM instance with 10 clusters
# fcm.fit(X)
# labeld_X = fcm.predict(X)                          # get the label of each data point

# transformed_X = fcm.centers[labeld_X]              # pixel quantization into the centers

# quatized_array = (                                 #Converting and saving image
#     transformed_X
#     .astype('uint8')                               # convert data points into 8-bit unsigned integers
#     .reshape((700, 700, 3))                            # reshape image
# )

# quatized_image = Image.fromarray(np.asarray(quatized_array))   # convert array into a PIL image object

# plt.subplot(5,3,15)
# plt.imshow(quatized_image)

plt.tight_layout()
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
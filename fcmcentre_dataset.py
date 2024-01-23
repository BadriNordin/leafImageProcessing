from msilib.schema import Directory
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
from fcmeans import FCM
from PIL import Image
from datetime import datetime

# start_time = datetime.now()

# dir = 'C:\\Users\\user\\OneDrive - Universiti Teknologi PETRONAS\\Documents\\UTP\\4th2nd\\FYP\\svm\\dataset3'

# categories = ['Healthy','Unhealthy']

# data = []

# for category in categories:
#     path = os.path.join(dir,category)
#     label = categories.index(category)

#     for img in os.listdir(path):
#         imgpath = os.path.join(path,img)
#         leaf_img = cv2.imread(imgpath)
        
#         #insert masking and clustering
#         leaf_img = cv2.resize(leaf_img,(700,700))
#         leaf_img = cv2.cvtColor(leaf_img, cv2.COLOR_RGB2HSV)
#         h,s,v = cv2.split(leaf_img)

#         #masks
#         # maskh0 = cv2.inRange(h,85,120)

#         # #applied masks
#         # leaf_imgf = cv2.bitwise_and(leaf_img,leaf_img,mask= maskh0)

#         #clustering
#         N = 700
#         M = 700

#         X = (
#         np.asarray(leaf_img)                              # convert a PIL image to np array
#         .reshape((N*M, 3))                             # reshape the image to convert each pixel to an instance of a data set
#         )

#         X = [i for i in X if i != 0]

#         #Creating and fitting the model
#         fcm = FCM(n_clusters=4)                           # create a FCM instance with 10 clusters
#         # fcm = FCM(random_state=1)
#         fcm.fit(X)

#         #Pixel quantization
#         labeld_X = fcm.predict(X)                          # get the label of each data point

#         transformed_X = fcm.centers[labeld_X]              # pixel quantization into the centers

#         #Converting and saving image
#         quatized_array = (
#             transformed_X
#             .astype('uint8')                               # convert data points into 8-bit unsigned integers
#             .reshape((M, N, 3))                            # reshape image
#         )

#         # quatized_image = Image.fromarray(np.asarray(quatized_array))   # convert array into a PIL image object
#         #end of clustering

#         nz = np.array(quatized_array).flatten()
        
#         #only take fcm.centres
#         nz2 = [i for i in nz if i != 0]

#         data.append([nz2,label])

# end_time = datetime.now()

# print('Duration: {}'.format(end_time - start_time))
# print(len(data))
# print(type(data))

# pick_in = open('fcmcentredataset04.pickle','wb')
# pickle.dump(data,pick_in)
# pick_in.close()

leaf_img = cv2.imread("C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Documents/UTP/4th2nd/FYP/Site/visit 1/unhealthy/IMG_1491.JPG")

#insert masking and clustering
leaf_img = cv2.resize(leaf_img,(700,700))
leaf_img = cv2.cvtColor(leaf_img, cv2.COLOR_RGB2HSV)
h,s,v = cv2.split(leaf_img)

#masks
maskh0 = cv2.inRange(h,85,120)

#applied masks
leaf_imgf = cv2.bitwise_and(leaf_img,leaf_img,mask= maskh0)

#clustering
N = 700
M = 700

X = (
np.asarray(leaf_imgf)                              # convert a PIL image to np array
 .reshape((N*M, 3))                               # reshape the image to convert each pixel to an instance of a data set
)

# X = [i for i in X if i != 0]

#Creating and fitting the model
fcm = FCM(n_clusters=8)                           # create a FCM instance with 10 clusters
# fcm = FCM(random_state=1)
fcm.fit(X)

#Pixel quantization
labeld_X = fcm.predict(X)                          # get the label of each data point

transformed_X = fcm.centers[labeld_X]              # pixel quantization into the centers

#Converting and saving image
quatized_array = (
    transformed_X
    .astype('uint8')                               # convert data points into 8-bit unsigned integers
    .reshape((M, N, 3))                            # reshape image
)

print('X type:',type(X))
print('fcmcentre:\n',fcm.centers)
print('array:\n',quatized_array)
from matplotlib.pyplot import axis, hsv
import numpy as np
from PIL import Image
from fcmeans import FCM
import sys 
import cv2
from datetime import datetime

start_time = datetime.now()

np.set_printoptions(threshold=np.inf)

# image = Image.open('C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Documents/UTP/4th2nd/FYP/Site/visit 1/unhealthy/IMG_1491.JPG')
# N, M = image.size    

image = cv2.imread("C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Documents/UTP/4th2nd/FYP/Site/visit 1/unhealthy/testcv2save.JPG")

masked = cv2.imread("C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Documents/UTP/4th2nd/FYP/Site/visit 1/testimages/testsaveres2.jpg")

dmask = cv2.imread("C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Documents/UTP/4th2nd/FYP/Site/visit 1/testimages/diseasemask.jpg")

hsvmask = cv2.imread("C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Documents/UTP/4th2nd/FYP/Site/visit 1/testimages/hsvmasked.jpg")

hmask = cv2.imread("C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Documents/UTP/4th2nd/FYP/Site/visit 1/testimages/hmasked.jpg")

hsvmaskh0 = cv2.imread("C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Documents/UTP/4th2nd/FYP/Site/visit 1/testimages/hsvmaskedh0.jpg")

hleaf = cv2.imread("C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Documents/UTP/4th2nd/FYP/Site/visit 1/testimages/healthy.jpg")

hleaf2 = cv2.imread("C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Documents/UTP/4th2nd/FYP/Site/visit 1/testimages/healthy2maskh0.jpg")

hhleaf2 = cv2.imread("C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Documents/UTP/4th2nd/FYP/Site/visit 1/testimages/hhealthy2maskh0.jpg")

M = 700
N = 700

image = hhleaf2

dim = (700,700)
image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

#Transforming image into a data set
X = (
    np.asarray(image)                              # convert a PIL image to np array
    .reshape((N*M, 3))                             # reshape the image to convert each pixel to an instance of a data set
)

#Creating and fitting the model
fcm = FCM(n_clusters=8)                           # create a FCM instance with 10 clusters
# fcm = FCM(random_state=1)
fcm.fit(X)

#Pixel quantization
labeld_X = fcm.predict(X)                          # get the label of each data point

# print("\nUnsorted:\n",fcm.centers)

sort = fcm.centers[fcm.centers[:,0].argsort()]
# fcm.centers = fcm.centers[fcm.centers[:,0].argsort()]

# print("\nSorted:")
# print(sort)

# sort[0] = [255,255,255]
# sort[1] = [0,0,0]
# sort[2] = [0,0,0]
# sort[3] = [0,0,0]
        

transformed_X = fcm.centers[labeld_X]              # pixel quantization into the centers
# transformed_X = sort[labeld_X]              # pixel quantization into the centers

#Converting and saving image
quatized_array = (
    transformed_X
    .astype('uint8')                               # convert data points into 8-bit unsigned integers
    .reshape((M, N, 3))                            # reshape image
)

quatized_image = Image.fromarray(np.asarray(quatized_array))   # convert array into a PIL image object
# quatized_image.save('C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Documents/UTP/4th2nd/FYP/Site/visit 1/testimages/hhealthy2maskh0random1_4.jpg') # save image

quatized_image.show()

# fcm.centers = fcm.centers[fcm.centers[:,0].argsort()]
# sort = fcm.centers[fcm.centers[:,0].argsort()]

# print("\nColours:")
# print(sort)

print("\nUnsorted2:\n",fcm.centers)

print('\nQuatized Array:\n',quatized_array)

# print("\nLabeld_X:")
# print(transformed_X,"\n")
# print(type(transformed_X))
end_time = datetime.now()
print('\nDuration: {}'.format(end_time - start_time))

import numpy as np
from PIL import Image
from fcmeans import FCM
import sys 
import cv2

# image = Image.open('C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Documents/UTP/4th2nd/FYP/Site/visit 1/idk/IMG_1491.JPG')
# N, M = image.size    

image = cv2.imread("C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Documents/UTP/4th2nd/FYP/Site/visit 1/idk/testcv2save.JPG")

masked = cv2.imread("C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Documents/UTP/4th2nd/FYP/Site/visit 1/testimages/testsaveres2.jpg")

M = 1280
N = 1920

image = masked

#Transforming image into a data set
X = (
    np.asarray(image)                              # convert a PIL image to np array
    .reshape((N*M, 3))                             # reshape the image to convert each pixel to an instance of a data set
)

#Creating and fitting the model
fcm = FCM(n_clusters=10)                           # create a FCM instance with 10 clusters
fcm.fit(X)

#Pixel quantization
labeld_X = fcm.predict(X)                          # get the label of each data point
fcm.centers[0] = [255,153,204]
fcm.centers[1] = [255,153,204]
fcm.centers[2] = [255,153,204]
fcm.centers[3] = [255,153,204]
fcm.centers[4] = [255,153,204]
# fcm.centers[5] = [255,153,204]
# fcm.centers[6] = [255,153,204]
# fcm.centers[7] = [255,153,204]
# fcm.centers[8] = [255,153,204]
# fcm.centers[9] = [255,153,204]

transformed_X = fcm.centers[labeld_X]              # pixel quantization into the centers

#Converting and saving image
quatized_array = (
    transformed_X
    .astype('uint8')                               # convert data points into 8-bit unsigned integers
    .reshape((M, N, 3))                            # reshape image
)

quatized_image = Image.fromarray(np.asarray(quatized_array))   # convert array into a PIL image object
quatized_image.save('C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Documents/UTP/4th2nd/FYP/Site/visit 1/testimages/testmaskpinaa.jpg') # save image

# #Final Compressed Image
# side_by_side = Image.fromarray(
#     np.hstack([
#         np.array(image),
#         np.array(quatized_image)
#     ])
# )
# side_by_side

# quatized_image = quatized_image.reshape((-1,3))
# quatized_image = quatized_image[transformed_X == fcm.centers[1]] = [255,255,255]
quatized_image.show()
print(fcm.centers)
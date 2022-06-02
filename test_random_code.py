from matplotlib.pyplot import axis
import numpy as np
from PIL import Image
from fcmeans import FCM
import sys 
import cv2

# image = Image.open('C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Documents/UTP/4th2nd/FYP/Site/visit 1/idk/IMG_1491.JPG')
# N, M = image.size    

image = cv2.imread("C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Documents/UTP/4th2nd/FYP/Site/visit 1/idk/testcv2save.JPG")

masked = cv2.imread("C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Documents/UTP/4th2nd/FYP/Site/visit 1/testimages/testsaveres2.jpg")

dmask = cv2.imread("C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Documents/UTP/4th2nd/FYP/Site/visit 1/testimages/diseasemask.jpg")
M = 1280
N = 1920

image = dmask

#Transforming image into a data set
X = (
    np.asarray(image)                              # convert a PIL image to np array
    .reshape((N*M, 3))                             # reshape the image to convert each pixel to an instance of a data set
)

#Creating and fitting the model
fcm = FCM(n_clusters=4)                           # create a FCM instance with 10 clusters
fcm.fit(X)

#Pixel quantization
labeld_X = fcm.predict(X)                          # get the label of each data point

print("unsorted:",fcm.centers)

sort = fcm.centers[fcm.centers[:,0].argsort()]

print("sorted:",sort)

fcm.centers = fcm.centers[fcm.centers[:,0].argsort()]

sort[0] = [0,0,0]
sort[1] = [0,0,0]
sort[2] = [0,0,0]

# for i in range(len(fcm.centers)):
#     for j in range(len(fcm.centers[i])):
#         if fcm.centers[i] == sort[0]:
#             fcm.centers[0] = [0,0,0]
#         elif fcm.centers[i] == sort[1]:
#             fcm.centers[1] = [0,0,0]
#         elif fcm.centers[i] == sort[2]:
#             fcm.centers[2] = [0,0,0]
        

# transformed_X = fcm.centers[labeld_X]              # pixel quantization into the centers
transformed_X = sort[labeld_X]              # pixel quantization into the centers

#Converting and saving image
quatized_array = (
    transformed_X
    .astype('uint8')                               # convert data points into 8-bit unsigned integers
    .reshape((M, N, 3))                            # reshape image
)

quatized_image = Image.fromarray(np.asarray(quatized_array))   # convert array into a PIL image object
quatized_image.save('C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Documents/UTP/4th2nd/FYP/Site/visit 1/testimages/disease4masked.jpg') # save image

# #Final Compressed Image
# side_by_side = Image.fromarray(
#     np.hstack([
#         np.array(image),
#         np.array(quatized_image)
#     ])
# )
# side_by_side

quatized_image.show()

# fcm.centers = fcm.centers[fcm.centers[:,0].argsort()]
# sort = fcm.centers[fcm.centers[:,0].argsort()]

print("\nfcm:")
print(sort)
# print("\nSorted:")
# print(sort)
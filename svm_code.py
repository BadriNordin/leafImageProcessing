from msilib.schema import Directory
import os
from pyexpat import model
import random
from statistics import mode
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
from fcmeans import FCM
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from datetime import datetime

start_time = datetime.now()

pick_in = open('hsv2.pickle','rb')
data = pickle.load(pick_in)
pick_in.close()

random.shuffle(data)
features = []
labels = []

for feature, label in data:
    features.append(feature)
    labels.append(label)

xtrain, xtest, ytrain,ytest = train_test_split(features, labels, test_size= 0.25 )

# model = SVC(C=1,kernel='poly',gamma='auto')
# model.fit(xtrain,ytrain)

pick = open('trained_hsv2.sav','rb')
# pickle.dump(model, pick)
model = pickle.load(pick)
pick.close()

prediction = model.predict(xtest)
accuracy = model.score(xtest, ytest)

categories = ['Healthy','Unhealthy']

end_time = datetime.now()

print('Duration: {}'.format(end_time - start_time))
print('Accuracy: ',accuracy)
print('Prediction: ',categories[prediction[0]])
print("Confusion Matrix:\n",confusion_matrix(prediction,ytest))

# leaf = xtest[0].reshape(700,700,3)
# leaf = Image.fromarray(np.asarray(leaf))
# leaf = xtest[0]
# print(type(leaf))
# plt.imshow(leaf)
# plt.show
# leaf.show()

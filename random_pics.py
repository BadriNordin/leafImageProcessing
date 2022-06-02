import cv2
import matplotlib.pyplot as plt

label16 = cv2.imread("C:/Users/user/OneDrive - Universiti Teknologi PETRONAS/Documents/UTP/4th2nd/FYP/Site/visit 1/testimages/testlabel16.jpg")
label16 = cv2.cvtColor(label16, cv2.COLOR_BGR2RGB)

plt.imshow(label16)

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

model = load_model('V1_828.h5')

X = []
for info in os.listdir(r'G:\\haihan\\Segmentation\\data\\test'):
    A = cv2.imread("data\\test\\" + info)
    X.append(A)
    # i += 1
X = np.array(X)
print(X.shape)
Y = model.predict(X)


groudtruth = []
for info in os.listdir(r'G:\\haihan\\Segmentation\\data\\test_groudtruth'):
    A = cv2.imread("data\\test_groudtruth\\" + info)
    groudtruth.append(A)
groudtruth = np.array(groudtruth)

a = range(10)
n = np.random.choice(a)
cv2.imwrite('prediction.png',Y[n])
cv2.imwrite('groudtruth.png',groudtruth[n])
fig, axs = plt.subplots(1, 3)
# cnt = 1
# for j in range(1):
axs[0].imshow(np.abs(X[n]))
axs[0].axis('off')
axs[1].imshow(np.abs(Y[n]))
axs[1].axis('off')
axs[2].imshow(np.abs(groudtruth[n]))
axs[2].axis('off')
    # cnt += 1
fig.savefig("imagestest.png")
plt.close()


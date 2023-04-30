import numpy as np
import glob
import cv2
import os
numImages = 30000

if numImages == 'max':
    train = np.zeros([len(os.listdir('./celeba/')), 32, 32, 3])
else:
    train = np.zeros([numImages, 32, 32, 3])

for count, imgPath in enumerate(glob.glob('./celeba/*jpg')):
    if numImages != 'max':
        if count >= numImages:
            break
    im = cv2.imread(imgPath)
    im = cv2.resize(im, (32,32))
    im = np.array(im)
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    train[count] = (im - 127.5) / 127.5
    print(count)

print(train.shape)
np.save('train', train)
#data_aug.py

import numpy as np
import os
import cv2
import random

from sklearn.decomposition import PCA

gaussian = np.random.normal(0, 0.01, 12)
dir = 'C:/Users/kevin/Documents/IISc/WCE/WCE_Dataset/VillousOedemas/data'
dir_mask = 'C:/Users/kevin/Documents/IISc/WCE/WCE_Dataset/VillousOedemas/annotations'

images = os.listdir(dir)
masks = os.listdir(dir_mask)

for im in images:
    img = cv2.imread(os.path.join(dir, im))
    mask_name = im.split('.')[0] + 'm.png'
    mask = cv2.imread(os.path.join(dir_mask, mask_name))

    img_reshape = np.reshape(img, (img.shape[0]*img.shape[1], 3))
    m = np.mean(img_reshape, axis=0)
    img_reshape = img_reshape - m

    pca = PCA(n_components=3)
    pca.fit(img_reshape)
    pc_vector = pca.components_
    pc_values = pca.explained_variance_

    for i in range(len(gaussian)):
        pc_values[0] *= random.choice(gaussian)
        pc_values[1] *= random.choice(gaussian)
        pc_values[2] *= random.choice(gaussian)
        val = np.sum(np.dot(pc_vector, pc_values.T))
        # print('val: ', val)
        img = img + val
        # img /= 255 
        cv2.imwrite(dir + '/eig_' + im, img)
        cv2.imwrite(dir_mask +'/eig_' + mask_name, mask)
        # cv2.imshow('img', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


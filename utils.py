from sklearn.metrics import confusion_matrix
import torch
from numba import jit
import os
import cv2
import numpy as np

def metrics(pred, target, threshold=0.8):

    pred = (pred > threshold).float()
    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()

    pred = pred.flatten()
    target = target.flatten()

    TN, FP, FN, TP = confusion_matrix(target, pred).ravel()

    SE = TP / (TP + FN)

    SPE = TN / (TN + FP)

    error_rate = (FP + FN) / (TP + FN + TN + FP)

    ACC = 1 - error_rate

    dice = (2 * TP) / (2 * TP + FP + FN)

    return SE, SPE, ACC, dice


class mean_std(object):

    def __init__(self, img_path, images, abnormality):
        super(mean_std, self).__init__()

        self.img_path = img_path
        self.images = images
        self.abnormality = abnormality

    def _read(self):
        if self.abnormality == 'polypoids':
            mean = np.zeros((1, 4))
            std = np.zeros((1, 4))
        else:
            mean = np.zeros((1, 3))
            std = np.zeros((1, 3))

        for i in range(len(self.images)):
            img = cv2.imread(os.path.join(self.img_path, self.images[i]))
            if self.abnormality == 'polypoids':
                img_cie = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                img_l, img_a, img_b = cv2.split(img_cie)
                img_a = np.expand_dims(img_a, axis=-1)
                img = np.concatenate((img, img_a), axis=-1)
            img = img/255
            m = np.mean(img, axis=(0,1))
            s = np.std(img, axis=(0,1))
            mean += m 
            std += s 
        mean = mean / i 
        std = std / i
        return mean, std

class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    @property
    def mloss(self):
        return self.avg

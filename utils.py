from sklearn.metrics import confusion_matrix
import torch
from numba import jit
import os
import cv2
import numpy as np
from PIL import Image 
import random 
from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    Compose,
    Transpose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
    CLAHE,
    OneOf,
    RandomBrightnessContrast,
    RandomGamma,
    ShiftScaleRotate
)

def metrics(pred, target, threshold=0.5):

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
        mean = np.zeros((1, 3))
        std = np.zeros((1, 3))

        for i in range(len(self.images)):
            img = cv2.imread(os.path.join(self.img_path, self.images[i]))
            img = cv2.resize(img, (448, 448), cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
            # print(img.shape)
            img = img/255
            m = np.mean(img, axis=(0,1))
            s = np.std(img, axis=(0,1))
            mean += m 
            std += s 
        mean = [x / i for x in mean]
        std = [x / i for x in std]
        return mean[0], std[0]        

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


class Augmentation(object):

    def __init__(self):
        super(Augmentation, self).__init__()

        self._hflip = HorizontalFlip(p=0.5)
        self._vflip = VerticalFlip(p=0.5)
        self._clahe = CLAHE(p=.5)
        self._rotate = RandomRotate90(p=.5)
        self._brightness = RandomBrightnessContrast(p=0.5)
        self._gamma = RandomGamma(p=0.5)
        self._transpose = Transpose(p=0.5)
        self._elastic = ElasticTransform(
            p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03)
        self._distort = GridDistortion(p=0.5)
        self._affine = ShiftScaleRotate(
            shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5)

        # self.aug_dict = {'hflip': self._hflip, 'vflip': self._vflip, 'rotate': self._rotate,
        #                  'elastic': self._elastic, 'distort': self._distort, 'transpose': self._transpose,
        #                  'clahe': self._clahe, 'bright': self._brightness, 'gamma': self._gamma,
        #                  'affine': self._affine
        #                  }

    def _aug(self):
        iter_max = 0
        aug = [self._hflip, self._vflip, self._clahe, self._rotate, self._brightness,
                self._gamma, self._transpose, self._elastic, self._distort, self._affine]
        # while iter_max < random.randint(4, 9):
        #     self.aug = random.choice(list(self.aug_dict.values()))
        #     aug.append(self.aug)
        #     iter_max += 1
        return Compose(aug)


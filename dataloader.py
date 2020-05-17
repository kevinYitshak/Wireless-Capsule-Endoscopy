# dataloader.py

import os
import random
from natsort import natsorted
import numpy as np
import cv2
from PIL import Image
from utils import mean_std, Augmentation

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
from torchvision.transforms import functional as F


class Angioectasias(Dataset):
    def __init__(self, abnormality, mode):
        super(Angioectasias, self).__init__()

        self.abnormality = abnormality
        self.mode = mode

        if self.mode == "train":
            self.img_path = "./" + self.abnormality + "/train/images"
            self.mask_path = "./" + self.abnormality + "/train/masks"

        if self.mode == "val" or self.mode == "test":
            self.img_path = "./" + self.abnormality + "/test/images"
            self.mask_path = "./" + self.abnormality + "/test/masks"

        self.images = natsorted(os.listdir(self.img_path))
        print(self.images)

        self.mean, self.std = mean_std(
            self.img_path, self.images, self.abnormality
        )._read()
        print("Mean: {}, Std: {}".format(self.mean, self.std))

        if self.mode == "train":
            self.aug = Augmentation()

        self._img = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )
        self._mask = transforms.Compose(
            [transforms.ToPILImage(), transforms.ToTensor()]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        img = cv2.imread(os.path.join(self.img_path, self.images[index]))
        img = cv2.resize(img, (448, 448), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

        mask_name = self.images[index].split(".")[0] + "m.png"
        mask = cv2.imread(os.path.join(self.mask_path, mask_name), 0)
        mask = cv2.resize(mask, (448, 448))

        if self.mode == "train":
            _aug = self.aug._aug()
            augmented = _aug(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)

        img = self._img(img)
        mask = self._mask(mask)

        return img, mask

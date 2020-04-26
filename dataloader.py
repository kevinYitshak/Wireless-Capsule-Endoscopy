#dataloader.py

import os
import random
from natsort import natsorted
import numpy as np
import cv2
from PIL import Image 

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
from torchvision.transforms import functional as F

class Angioectasias(Dataset):

    def __init__(self, abnormality, mode):
        super(Angioectasias, self).__init__()

        self.abnormality = abnormality
        self.mode = mode

        if self.mode == 'train':
            self.img_path = './' + self.abnormality + '/train/images'
            self.mask_path = './' + self.abnormality + '/train/masks'

        if self.mode == 'val' or self.mode == 'test':
            self.img_path = './' + self.abnormality + '/test/images'
            self.mask_path = './' + self.abnormality + '/test/masks'

        self.images = natsorted(os.listdir(self.img_path))
        print(self.images)

        if self.mode == 'train' or self.mode == 'val':
            self._pil = transforms.ToPILImage()
            self._jitter = transforms.ColorJitter(brightness=.05, contrast=.05)
            self._grayscale = transforms.RandomGrayscale(p=0.3)
            self._rotate = transforms.RandomRotation(
                degrees=10, resample=Image.BICUBIC)
            self._tensor = transforms.ToTensor()
            self._norm = transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
        else:
            self._pil = transforms.ToPILImage()
            self._tensor = transforms.ToTensor()
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
            ])
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        img = cv2.imread(os.path.join(self.img_path, self.images[index]))
        img = cv2.resize(img, (448, 448))

        if self.abnormality == 'polypoids':
            img_cie = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            img_l, img_a, img_b = cv2.split(img_cie)
            img_a = np.expand_dims(img_a, axis=-1)
            img = np.concatenate((img, img_a), axis=-1)

        img = img.astype('uint8')


        mask_name = self.images[index].split('.')[0] + 'm.png'
        mask = cv2.imread(os.path.join(self.mask_path, mask_name), 0)
        mask = cv2.resize(mask, (448, 448))
        _, mask = cv2.threshold(mask, 30, 255, cv2.THRESH_BINARY)

        if self.mode == 'train' and self.mode == 'val':
            # 1. to PIL
            img = self._pil(img)
            mask = self._pil(mask)

            # 2. augmentation
            if 0.2 < random.random() < 0.8:
                img = self._jitter(img)
                img = self._grayscale(img)
                img = self._rotate(img)
                mask = self._rotate(mask)

            if random.random() < 0.5:
                img = F.hflip(img)
                mask = F.hflip(mask)
            
            if random.random() < 0.5:
                img = F.vflip(img)
                mask = F.vflip(mask)
        
            img = self._tensor(img)
            img = self._norm(img)
            mask = self._tensor(mask)

        else:
            img = self.transform(img)
            mask = self._pil(mask)
            mask = self._tensor(mask)

        return img, mask

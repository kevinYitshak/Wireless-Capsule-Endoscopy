#dataloader.py

import os
import random
from natsort import natsorted
import numpy as np
import cv2

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils

class Angioectasias(Dataset):

    def __init__(self, abnormality):
        super(Angioectasias, self).__init__()

        self.abnormality = abnormality
        self.img_path = './' + self.abnormality + '/train/images'
        self.mask_path = './' + self.abnormality + '/train/masks'
        
        self.images = natsorted(os.listdir(self.img_path))
        print(self.images)

        self.transform = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.Resize((448, 448)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]),
                        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        img = cv2.imread(os.path.join(self.img_path, self.images[index]))
        img = cv2.resize(img, (448, 448))
        # img_cie = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        # img_l, img_a, img_b = cv2.split(img_cie)
        # img_a = np.expand_dims(img_a, axis=-1)
        # img = np.concatenate((img, img_a), axis=-1)
        img = img.astype('uint8')
        img = self.transform(img)

        mask_name = self.images[index].split('.')[0] + 'm.png'
        mask = cv2.imread(os.path.join(self.mask_path, mask_name),0)
        mask = cv2.resize(mask, (448, 448))
        _, mask = cv2.threshold(mask, 30, 255, cv2.THRESH_BINARY)
        mask = np.expand_dims(mask, axis=-1)
        mask = np.transpose(mask, (2, 0, 1))
        mask[mask == 255] = 1
        mask = torch.from_numpy(mask).float()
        
        return img, mask

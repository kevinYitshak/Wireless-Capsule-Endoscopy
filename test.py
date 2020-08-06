#test.py

import argparse
import cv2
import numpy as np
import os
from natsort import natsorted
from tqdm import tqdm
from utils import AverageMeter, metrics, mean_std, Augmentation

import torch 
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.nn as nn

from models import U_Net, R2U_Net, AttU_Net, R2AttU_Net, Models
from dataloader import Angioectasias

class test_class(object):

    def __init__(self, abnormality):
        self.abnormality = abnormality
        self._init_args()
        self._init_device()
        self._init_dataset()
        self._init_model()

    def _init_args(self):

        parser = argparse.ArgumentParser(description='config')
        parser.add_argument('--mgpu', default=False,
                            help='Set true to use multi GPUs')

        self.args = parser.parse_args()


    def _init_device(self):

        # os.environ['CUDA_VISIBLE_DEVICES'] = '0'

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _init_dataset(self):
        
        M = Models()

        if self.args.mgpu:
            self.batch_size = 28
            print('batch_size: ', self.batch_size)
            self.date = '/2020-05-06~11:38:23'
            self.Mo = M.FPN(img_ch=3, output_ch=1)
        else: 
            self.batch_size = 7
            print('batch_size: ', self.batch_size)
            self.date = '/2020-05-25~05:51:58'
            self.Mo = U_Net(img_ch=3, output_ch=1)

        test_images = Angioectasias(self.abnormality, mode='test')
        self.test_queue = DataLoader(test_images, batch_size=self.batch_size, drop_last=False)

    def _init_model(self):
        
        if torch.cuda.device_count() > 1 and self.args.mgpu:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(self.Mo)
        else:
            model = self.Mo

        self.model = model.to(self.device)
    
    def mask_color_img(self, img, mask, tar, alpha=0.3):
        '''
        img: cv2 image
        mask: bool or np.where
        color: BGR triplet [_, _, _]. Default: [0, 255, 255] is yellow.
        alpha: float [0, 1]. 

        Ref: http://www.pyimagesearch.com/2016/03/07/transparent-overlays-with-opencv/
        '''
        out = img.copy()
        img_layer = img.copy()
        img_layer1 = img.copy()
        img_layer[mask == 255] = [0, 0, 255]
        img_layer1[tar == 255] = [0, 255, 0]
        out1 = cv2.addWeighted(img_layer, alpha, out, 1 - alpha, 0, out)
        out = cv2.addWeighted(img_layer1, alpha, out1, 1 - alpha, 0, out)
        return(out)

    def test(self):
        test_path = './' + abnormality + '/test/images'
        input_files = natsorted(os.listdir(test_path))
        save_path = './' + abnormality + '/pred/'

        self.mean, self.std = mean_std(test_path, input_files, abnormality)._read()

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        self.model.load_state_dict(torch.load('./' + self.abnormality + self.date \
            + '/ckpt/best_weights.pth.tar')['state_dict'])
        self.model.eval()

        self.test_dice = AverageMeter()
        self.test_accuracy = AverageMeter()
        self.test_sensitivity = AverageMeter()
        self.test_specificity = AverageMeter()

        self.tf = transforms.Compose([
            transforms.Normalize((-self.mean/self.std), (1/self.std))
        ])

        with torch.no_grad():
            k = 0
            for _, (image, target1) in enumerate(tqdm(self.test_queue)):

                image = image.to(self.device, dtype=torch.float32)
                target = self.model(image)
                target = torch.sigmoid(target)

                SE, SPE, ACC, DICE = metrics(target, target1)

                self.test_accuracy.update(ACC, image.size(0))
                self.test_sensitivity.update(SE, image.size(0))
                self.test_specificity.update(SPE, image.size(0))
                self.test_dice.update(DICE, image.size(0))

                for i in range(image.shape[0]):
                    img = image[i, :, :, :].detach().cpu()
                    img = self.tf(img).numpy()
                    img = np.transpose(img, (1, 2, 0))
                    img = img * 255
                    img = img.astype('uint8')

                    tar = target1[i, :, :, :].detach().cpu().numpy()
                    tar = np.transpose(tar, (1, 2, 0))
                    tar = tar * 255
                    tar.astype('uint8')
                    tar = np.squeeze(tar, axis=-1)

                    pred = target[i, :, :, :].detach().cpu().numpy()
                    pred = np.transpose(pred, (1, 2, 0))
                    pred = pred > 0.5
                    pred = pred * 255
                    pred.astype('uint8')
                    pred = np.squeeze(pred, axis=-1)

                    out = self.mask_color_img(img, pred, tar)

                    cv2.imwrite(save_path + input_files[k+i], out)
                k += self.batch_size

        print('Acc: {:.4f}, Sen: {:.4f}, Spe: {:.4f}, Dice: {:.4f}'\
            .format(self.test_accuracy.mloss,
                    self.test_sensitivity.mloss,
                    self.test_specificity.mloss,
                    self.test_dice.mloss))
    

if __name__ == '__main__':

    abnormality = 'Lymphangectasias'
    test = test_class(abnormality)
    test.test()  

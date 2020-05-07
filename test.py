#test.py

import argparse
import cv2
import numpy as np
import os
from natsort import natsorted
from tqdm import tqdm
from utils import AverageMeter, metrics, get_gpus_memory_info

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
        self._args()
        self._init_device()
        self._init_model()
        self._init_dataset()

    def _args(self):

        parser = argparse.ArgumentParser(description='config')
        parser.add_argument('--mgpu', default=False,
                            help='Set true to use multi GPUs')

        self.args = parser.parse_args()


    def _init_device(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _init_dataset(self):

        if self.args.mgpu:
            self.batch_size = 28
        else:
            self.batch_size = 7

        test_images = Angioectasias(self.abnormality, mode='test')
        self.test_queue = DataLoader(test_images, batch_size=self.batch_size, drop_last=False, num_workers=4)

    def _init_model(self):

        M = Models()
        model = M.FPN(img_ch=3, output_ch=1)
        
        if torch.cuda.device_count() > 1 and self.args.mgpu:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)

        self.model = model.to(self.device)
    
    def test(self):
        test_path = './' + abnormality + '/test/images'
        input_files = natsorted(os.listdir(test_path))
        save_path = './' + abnormality + '/pred/'

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        self.model.load_state_dict(torch.load('./' + self.abnormality + '/2020-04-30~07:18:12' \
            + '/ckpt/best_weights.pth.tar')['state_dict'])
        self.model.eval()

        self.test_dice = AverageMeter()
        self.test_accuracy = AverageMeter()
        self.test_sensitivity = AverageMeter()
        self.test_specificity = AverageMeter()

        with torch.no_grad():
            for k, (img, target) in enumerate(tqdm(self.test_queue)):

                img = img.to(self.device, dtype=torch.float32)
                out = self.model(img)

                out = torch.sigmoid(out)
                SE, SPE, ACC, DICE = metrics(out, target)

                self.test_accuracy.update(ACC, img.size(0))
                self.test_sensitivity.update(SE, img.size(0))
                self.test_specificity.update(SPE, img.size(0))
                self.test_dice.update(DICE, img.size(0))

                out = out[0].cpu().numpy()
                out = np.transpose(out, (1, 2, 0))
                out = out * 255
                out.astype('uint8')
                cv2.imwrite(save_path + input_files[k], out)
        
        print('Acc: {:.4f}, Sen: {:.4f}, Spe: {:.4f}, Dice: {:.4f}'\
            .format(self.test_accuracy.mloss,
                    self.test_sensitivity.mloss,
                    self.test_specificity.mloss,
                    self.test_dice.mloss))

if __name__ == '__main__':

    abnormality = 'vascular'
    test = test_class(abnormality)
    test.test()  

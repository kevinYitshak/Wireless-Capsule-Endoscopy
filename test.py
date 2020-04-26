#test.py

import cv2
import numpy as np
import os
from natsort import natsorted
from tqdm import tqdm
from utils import AverageMeter, metrics 

import torch 
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
import torchvision.transforms as transforms

from models import U_Net, R2U_Net, AttU_Net, R2AttU_Net
from dataloader import Angioectasias

class test_class(object):

    def __init__(self, abnormality):
        self.abnormality = abnormality
        self._init_device()
        self._init_model()
        self._init_dataset()

    def _init_device(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

        if not torch.cuda.is_available():
            print('GPU not available!')
            self.device = 'cpu'
        else:
            print('GPU is available!')
            cudnn.enabled = True
            cudnn.benchmark = True
            self.device = torch.device('cuda:{}'.format(0))

    def _init_dataset(self):

        test_images = Angioectasias(self.abnormality, mode='test')
        self.test_queue = DataLoader(test_images, batch_size=1, drop_last=False)

    def _init_model(self):

        if self.abnormality == 'polypoids':
            model = AttU_Net(img_ch=4, output_ch=1)
        else:
            model = AttU_Net(img_ch=3, output_ch=1)
        self.model = model.to(self.device)
    
    def test(self):
        test_path = './' + abnormality + '/test/images'
        input_files = natsorted(os.listdir(test_path))
        save_path = './' + abnormality + '/pred/'

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        self.model.load_state_dict(torch.load('./' + self.abnormality + '/2020-04-26~10:31:50' \
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

                self.test_accuracy.update(ACC, input.size(0))
                self.test_sensitivity.update(SE, input.size(0))
                self.test_specificity.update(SPE, input.size(0))
                self.test_dice.update(DICE, input.size(0))

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

    abnormality = 'ampulla-of-vater'
    test = test_class(abnormality)
    test.test()  

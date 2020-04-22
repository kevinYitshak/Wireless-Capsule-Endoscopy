#train.py

import os
import cv2
import numpy as np
from tqdm import tqdm
from datetime import datetime

import torch
import torchvision
import torch.nn as nn
from torch.utils import data
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchsummary import summary
from torch import optim
from tensorboardX import SummaryWriter
# from torch.utils.tensorboard import SummaryWriter

from dataloader import Angioectasias
#from Unet_Angioectasias import UNet
from models import U_Net, R2U_Net, AttU_Net, R2AttU_Net
from loss import DiceLoss
from dice_loss import dice_coeff
# from util.utils import store_images

def weights_init(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)

class wce_angioectasias(object):

    def __init__(self, abnormality):
        super(wce_angioectasias, self).__init__()

        self.abnormality = abnormality
        self._init_logger()
        self._init_device()
        self._init_dataset()
        self._init_model()

    def _init_logger(self):

        self.d = datetime.now().strftime('%Y-%m-%d~%H:%M:%S')
        self.path = './' + self.abnormality + '/'+ self.d

        os.makedirs(self.path + '/ckpt')
        os.makedirs(self.path + '/log')
        self.save_tbx_log = self.path + '/log'

        self.writer = SummaryWriter(self.save_tbx_log)

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

        train_img = Angioectasias(self.abnormality)
        num_train = len(train_img)
        indices = list(range(num_train))
        split = int(np.floor(0.9 * num_train))
        self.batch_size = 2
        self.train_queue = data.DataLoader(train_img, batch_size=self.batch_size,
                            drop_last=False,
                            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]))

        self.val_queue = data.DataLoader(train_img, batch_size=self.batch_size,
                            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]))

    def _init_model(self):

        criterion = nn.BCELoss() #nn.BCEWithLogitsLoss()
        self.criterion = criterion.to(self.device)

        model = AttU_Net(img_ch=3, output_ch=1)
        self.model = model.to(self.device)   
#        self.model.apply(weights_init)
        # summary(self.model, input_size=(4, 448, 448))
        self.model_optimizer = optim.Adamax(model.parameters(), lr=2e-3)

    def run(self):

        self.end_epoch = 30
        
        self.best_score = 0
        self.val_loss_meter = AverageMeter()
        self.val_dice_coeff_meter = AverageMeter()
        self.train_loss_meter = AverageMeter()

        for epoch in range(self.end_epoch):
            
            self.epoch = epoch
            print('Epoch: %d/%d' % (self.epoch + 1, self.end_epoch))

            self.train()

            self.val()

            self.val_loss_meter.reset()
            self.train_loss_meter.reset()
            self.val_dice_coeff_meter.reset()

        self.writer.close()

    def train(self):

        self.model.train()
        tbar = tqdm(self.train_queue)
        for step, (input, target) in enumerate(tbar):

            input = input.to(device=self.device, dtype=torch.float32)
            target = target.to(device=self.device, dtype=torch.float32)
            
            predicts = self.model(input)
            predicts = torch.sigmoid(predicts)
            self.dice = DiceLoss()
            self.loss = (0.75 * self.criterion(predicts, target) 
                        + 0.25 * self.dice(predicts, target))

            self.train_loss_meter.update(self.loss.item(), input.size(0))

            self.model_optimizer.zero_grad()
            self.loss.backward()
            self.model_optimizer.step()

            # print('Epoch_loss: ', self.train_loss_meter.mloss)
            tbar.set_description('train loss: %.4f' % (self.train_loss_meter.mloss))

        self.writer.add_scalar('Train/loss', self.train_loss_meter.mloss, self.epoch)
        print('Total_parameters: ', torch.sum(list(self.model.parameters())[0]))

    def val(self):

        self.model.eval()
        tbar = tqdm(self.val_queue)
        
        for step, (input, target) in enumerate(tbar):

            input = input.to(device=self.device, dtype=torch.float32)
            target = target.to(device=self.device, dtype=torch.float32)

            pred = self.model(input)

            pred = torch.sigmoid(pred)
            self.loss = self.criterion(pred, target)
            # self.dice = dice_coeff(pred, target.squeeze(dim=1))
            pred = (pred > .5).float()
            self.dice_score = 1 - self.dice(pred, target)
            self.val_loss_meter.update(self.loss.item(), input.size(0))
            self.val_dice_coeff_meter.update(self.dice_score.item(), input.size(0))
            
            tbar.set_description('Val_Loss: {}; Val_Dice: {}'.format(self.val_loss_meter.mloss, self.val_dice_coeff_meter.mloss))

            self.writer.add_images('Images', input, self.epoch)
            self.writer.add_images('Masks/True', target, self.epoch)
            self.writer.add_images('Masks/pred', pred, self.epoch)

        if self.dice_score > self.best_score:
            ckpt_file_path = self.path + '/ckpt/ckpt_{}.pth.tar'.format(self.epoch+1)
            torch.save(
                {
                'epoch': self.epoch,
                'state_dict': self.model.state_dict(),
                }, ckpt_file_path)

        self.writer.add_scalar('Val/Loss', self.val_loss_meter.mloss, self.epoch)
        self.writer.add_scalar('Val/Dice', self.val_dice_coeff_meter.mloss, self.epoch)
    
class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    @property
    def mloss(self):
        return self.avg

if __name__ == '__main__':

    abnormality = ['ampulla-of-vater'] #'polypoids', 'vascular', 'inflammatory']

    for name in abnormality:               
        train_network = wce_angioectasias(name)
        train_network.run()

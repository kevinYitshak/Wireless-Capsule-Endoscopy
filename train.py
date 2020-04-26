#train.py

import os
import cv2
import numpy as np
from tqdm import tqdm
from datetime import datetime
from utils import AverageMeter, metrics
from dataloader import Angioectasias

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
from torch.nn import init

from models import U_Net, R2U_Net, AttU_Net, R2AttU_Net
from loss import DiceLoss

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(
                    'initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

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

        train_img = Angioectasias(self.abnormality, mode='train')
        val_img = Angioectasias(self.abnormality, mode='val')
        # num_train = len(train_img)
        # indices = list(range(num_train))
        # split = int(np.floor(0.9 * num_train))
        self.batch_size = 2
        self.train_queue = data.DataLoader(train_img, batch_size=self.batch_size,
                            drop_last=False, shuffle=True)

        self.val_queue = data.DataLoader(val_img, batch_size=self.batch_size, shuffle=True)

    def _init_model(self):

        criterion = nn.BCELoss() #nn.BCEWithLogitsLoss()
        self.criterion = criterion.to(self.device)

        if self.abnormality == 'polypoids':
            model = AttU_Net(img_ch=4, output_ch=1)
        else:
            model = AttU_Net(img_ch=3, output_ch=1)

        self.model = model.to(self.device)
        init_weights(self.model, 'kaiming', gain=0.02)
        # summary(self.model, input_size=(4, 448, 448))
        self.model_optimizer = optim.Adam(model.parameters(), lr=1e-3)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.model_optimizer, T_max=len(self.train_queue))

    def run(self):

        self.end_epoch = 100
        
        self.best_score = 0
        #val_meter
        self.val_loss_meter = AverageMeter()
        self.val_dice = AverageMeter()
        self.val_accuracy = AverageMeter()
        self.val_sensitivity = AverageMeter()
        self.val_specificity = AverageMeter()

        #train_meter
        self.train_loss_meter = AverageMeter()
        self.train_accuracy = AverageMeter()
        self.train_sensitivity = AverageMeter()
        self.train_specificity = AverageMeter()
        self.tr_dice = AverageMeter()

        for epoch in range(self.end_epoch):
            
            self.epoch = epoch
            print('Epoch: %d/%d' % (self.epoch + 1, self.end_epoch))

            self.train()
            self.scheduler.step()
            print('Decay LR: ', self.scheduler.get_lr())
            
            #train
            self.train_loss_meter.reset()
            self.train_accuracy.reset()
            self.train_sensitivity.reset()
            self.train_specificity.reset()
            self.tr_dice.reset()

            self.val()

            #val
            self.val_loss_meter.reset()
            self.val_dice.reset()
            self.val_accuracy.reset()
            self.val_sensitivity.reset()
            self.val_specificity.reset()

        self.writer.close()

        print('Best_Dice: {:.4f}, Best_Sen: {:.4f}, Best_epoch: {}' \
            .format(self.best_dice, self.best_sen, self.best_epoch))

    def train(self):

        self.model.train()
        tbar = tqdm(self.train_queue)
        for step, (input, target) in enumerate(tbar):

            input = input.to(device=self.device, dtype=torch.float32)
            target = target.to(device=self.device, dtype=torch.float32)
            
            predicts = self.model(input)
            predicts_prob = torch.sigmoid(predicts)
            self.dice = DiceLoss()
            self.loss = (.75 * self.criterion(predicts_prob, target)
                        + .25 * self.dice(predicts_prob, target))

            self.train_loss_meter.update(self.loss.item(), input.size(0))

            self.model_optimizer.zero_grad()
            self.loss.backward()
            self.model_optimizer.step()

            ###########CAL METRIC############
            SE, SPE, ACC, DICE = metrics(predicts, target)

            self.train_accuracy.update(ACC, input.size(0))
            self.train_sensitivity.update(SE, input.size(0))
            self.train_specificity.update(SPE, input.size(0))
            self.tr_dice.update(DICE, input.size(0))
            #################################

            tbar.set_description('train loss: %.4f' % (self.train_loss_meter.mloss))

        self.writer.add_scalar('Train/loss', self.train_loss_meter.mloss, self.epoch)
        self.writer.add_scalar('Train/Acc', self.train_accuracy.mloss, self.epoch)
        self.writer.add_scalar('Train/Sen', self.train_sensitivity.mloss, self.epoch)
        self.writer.add_scalar('Train/Spe', self.train_specificity.mloss, self.epoch)
        self.writer.add_scalar('Train/Dice', self.tr_dice.mloss, self.epoch)

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
            # pred = (pred > .5).float()
            # self.dice_score = 1 - self.dice(pred, target)
            self.val_loss_meter.update(self.loss.item(), input.size(0))

            ###########CAL METRIC############
            SE, SPE, ACC, DICE = metrics(pred, target)

            self.val_accuracy.update(ACC, input.size(0))
            self.val_sensitivity.update(SE, input.size(0))
            self.val_specificity.update(SPE, input.size(0))
            self.val_dice.update(DICE, input.size(0))
            #################################
            tbar.set_description('Val_Loss: {:.4f}; Val_Dice: {:.4f}'.format(self.val_loss_meter.mloss, self.val_dice.mloss))

            self.writer.add_images('Images', input, self.epoch)
            self.writer.add_images('Masks/True', target, self.epoch)
            self.writer.add_images('Masks/pred', pred, self.epoch)

        if (self.val_dice.mloss + self.val_sensitivity.mloss) > self.best_score:
            self.best_score = self.val_dice.mloss + self.val_sensitivity.mloss
            self.best_dice = self.val_dice.mloss
            self.best_sen = self.val_sensitivity.mloss
            self.best_epoch = self.epoch

            ckpt_file_path = self.path + '/ckpt/best_weights.pth.tar'
            torch.save(
                {
                'epoch': self.epoch,
                'state_dict': self.model.state_dict(),
                }, ckpt_file_path)

        self.writer.add_scalar('Val/Loss', self.val_loss_meter.mloss, self.epoch)
        self.writer.add_scalar('Val/Dice', self.val_dice.mloss, self.epoch)
        self.writer.add_scalar('Val/Acc', self.val_accuracy.mloss, self.epoch)
        self.writer.add_scalar('Val/Sen', self.val_sensitivity.mloss, self.epoch)
        self.writer.add_scalar('Val/Spe', self.val_specificity.mloss, self.epoch)

if __name__ == '__main__':

    abnormality = ['ampulla-of-vater'] #'polypoids', 'vascular', 'inflammatory']

    for name in abnormality:               
        train_network = wce_angioectasias(name)
        train_network.run()

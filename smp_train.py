import os
from datetime import datetime

import argparse
import segmentation_models_pytorch as smp
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torch import optim
from torch.nn import init
from torch.utils import data

from dataloader import Angioectasias
from models import Models, U_Net, R2U_Net, AttU_Net, R2AttU_Net
from utils import AverageMeter, metrics


class wce_angioectasias:
    def __init__(self, abnormality):
        super(wce_angioectasias, self).__init__()

        self.abnormality = abnormality
        self._args()
        self._init_logger()
        self._init_device()
        self._init_dataset()
        self._init_model()
        self._init_params()

    def _args(self):

        parser = argparse.ArgumentParser(description="config")
        parser.add_argument("--mgpu", default=False, help="Set true to use multi GPUs")

        self.args = parser.parse_args()

    def _init_logger(self):

        self.d = datetime.now().strftime("%Y-%m-%d~%H:%M:%S")
        self.path = "./" + self.abnormality + "/" + self.d

        if not os.path.exists(self.path + "/ckpt"):
            os.makedirs(self.path + "/ckpt")
            os.makedirs(self.path + "/log")
        self.save_tbx_log = self.path + "/log"

        self.writer = SummaryWriter(self.save_tbx_log)

    def _init_device(self):

        # os.environ['CUDA_VISIBLE_DEVICES'] = '0'

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _init_dataset(self):

        train_img = Angioectasias(self.abnormality, mode="train")
        val_img = Angioectasias(self.abnormality, mode="val")
        if self.args.mgpu:
            self.batch_size = 28
        else:
            self.batch_size = 7
        self.train_queue = data.DataLoader(
            train_img,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=True,
            num_workers=4,
        )

        self.val_queue = data.DataLoader(
            val_img, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def _init_model(self):

        M = Models()
        model = M.FPN(img_ch=3, output_ch=1)

        if torch.cuda.device_count() > 1 and self.args.mgpu:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)

        self.model = model.to(self.device)

    def _init_params(self):

        self.end_epoch = 10
        self.loss_bce = smp.utils.losses.BCEWithLogitsLoss()
        self.loss_dice = smp.utils.losses.DiceLoss(activation="sigmoid")
        self.loss = smp.utils.losses.base.SumOfLosses(self.loss_bce, self.loss_dice)
        self.loss = smp.utils.losses.base.SumOfLosses(self.loss_bce, self.loss_dice)

        self.metrics = [
            smp.utils.metrics.IoU(activation="sigmoid"),
            smp.utils.metrics.Fscore(activation="sigmoid"),
            smp.utils.metrics.Recall(activation="sigmoid"),
        ]
        self.optimizer = torch.optim.Adamax(
            [dict(params=self.model.parameters(), lr=1e-3),]
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=len(self.train_queue)
        )

    def _run(self):

        self.train_epoch = smp.utils.train.TrainEpoch(
            self.model,
            loss=self.loss,
            metrics=self.metrics,
            optimizer=self.optimizer,
            device=self.device,
            verbose=True,
        )

        self.valid_epoch = smp.utils.train.ValidEpoch(
            self.model,
            loss=self.loss,
            metrics=self.metrics,
            device=self.device,
            verbose=True,
        )

        self.best_dice = 0

        for epoch in range(0, self.end_epoch):

            self.epoch = epoch
            print("----- Epoch: %d/%d -----" % (self.epoch + 1, self.end_epoch))

            self._train()

            self.scheduler.step()
            # print('Decay LR: ', self.scheduler.get_lr())

            self._val()

        self.writer.close()

        print(
            "Best_Dice: {:.4f}, Best_Sen: {:.4f}, Best_epoch: {}".format(
                self.best_dice, self.best_sen, self.best_epoch
            )
        )

    def _train(self):

        train_logs, image, target, pred = self.train_epoch.run(self.train_queue)
        train_logs, image, target, pred = self.train_epoch.run(self.train_queue)
        # print(train_logs)

        self.writer.add_images("Train/Images", image, self.epoch)
        self.writer.add_images("Train/Masks/True", target, self.epoch)
        self.writer.add_images("Train/Masks/pred", (pred > 0.5).float(), self.epoch)

        self.writer.add_scalar(
            "Train/loss", train_logs["bce_with_logits_loss + dice_loss"], self.epoch
        )
        self.writer.add_scalar("Train/Sen", train_logs["sensitivity"], self.epoch)
        self.writer.add_images("Train/Masks/pred", (pred > 0.5).float(), self.epoch)

        self.writer.add_scalar(
            "Train/loss", train_logs["bce_with_logits_loss + dice_loss"], self.epoch
        )
        self.writer.add_scalar("Train/Sen", train_logs["sensitivity"], self.epoch)
        self.writer.add_scalar("Train/IOU", train_logs["iou"], self.epoch)
        self.writer.add_scalar("Train/Dice", train_logs["dice"], self.epoch)

    def _val(self):

        valid_logs, image, target, pred = self.valid_epoch.run(self.val_queue)
        # print(valid_logs)

        self.writer.add_images("Val/Images", image, self.epoch)
        self.writer.add_images("Val/Masks/True", target, self.epoch)
        self.writer.add_images("Val/Masks/pred", (pred > 0.5).float(), self.epoch)

        self.writer.add_scalar(
            "Val/loss", valid_logs["bce_with_logits_loss + dice_loss"], self.epoch
        )
        self.writer.add_scalar("Val/Sen", valid_logs["sensitivity"], self.epoch)
        self.writer.add_images("Val/Masks/pred", (pred > 0.5).float(), self.epoch)

        self.writer.add_scalar(
            "Val/loss", valid_logs["bce_with_logits_loss + dice_loss"], self.epoch
        )
        self.writer.add_scalar("Val/Sen", valid_logs["sensitivity"], self.epoch)
        self.writer.add_scalar("Val/IOU", valid_logs["iou"], self.epoch)
        self.writer.add_scalar("Val/Dice", valid_logs["dice"], self.epoch)

        # do something (save model, change lr, etc.)
        if self.best_dice < valid_logs["dice"]:
            self.best_dice = valid_logs["dice"]
            self.best_sen = valid_logs["sensitivity"]
            self.best_epoch = self.epoch

            ckpt_file_path = self.path + "/ckpt/best_weights.pth.tar"
            torch.save(
                # {"epoch": self.epoch, "state_dict": self.model.state_dict(), },
                # ckpt_file_path
                {"epoch": self.epoch, "state_dict": self.model.state_dict(),},
                ckpt_file_path,
            )


if __name__ == "__main__":

    # kid2 = ['polypoids', 'vascular', 'ampulla-of-vater', 'inflammatory']
    kid1 = [
        "Angioectasias",
        "Apthae",
        "Bleeding",
        "ChylousCysts",
        "Lymphangectasis",
        "Polypoids",
        "Stenoses",
        "Ulcers",
    ]

    # abnormality = ['Bleeding']

    for name in kid1:
        print("*" * 50)
        print(name)
        print("*" * 50)
        train_network = wce_angioectasias(name)
        train_network._run()

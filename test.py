#test.py

import cv2
import numpy as np
import os
from natsort import natsorted
from tqdm import tqdm

import torch 
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
import torchvision.transforms as transforms

from models import U_Net, R2U_Net, AttU_Net, R2AttU_Net
from dataloader import Angioectasias


class ReadImages(Dataset):

        def __init__(self, path):
            super(ReadImages).__init__()

            self.path = path
            self.images = natsorted(os.listdir(self.path))
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

        def __len__(self):
            return len(self.images)

        def __getitem__(self, index):
            img = cv2.imread(os.path.join(self.path, self.images[index]))
            img = cv2.resize(img, (448, 448), interpolation=cv2.INTER_CUBIC)
            # img_cie = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            # img_l, img_a, img_b = cv2.split(img_cie)
            # img_a = np.expand_dims(img_a, axis=-1)
            # img = np.concatenate((img, img_a), axis=-1)
            img = img.astype('uint8')
            img = self.transform(img)
            return img

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

        model = AttU_Net(img_ch=3, output_ch=1)
        self.model = model.to(self.device)
    
    def test(self):
        test_path = './' + abnormality + '/train/images'
        input_files = natsorted(os.listdir(test_path))
        save_path = './' + abnormality + '/pred/'

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        self.model.load_state_dict(torch.load('./' + self.abnormality + '/ckpt/ckpt_2.pth.tar')['state_dict'])
        self.model.eval()

        with torch.no_grad():
            for k, img in enumerate(tqdm(self.test_queue)):

                img = img.to(self.device, dtype=torch.float32)
                out = self.model(img)

                out = torch.sigmoid(out)
                out = (out).float()

                out = out[0].cpu().numpy()
                out = np.transpose(out, (1, 2, 0))
                out = out * 255
                out.astype('uint8')
                cv2.imwrite(save_path + input_files[k], out)
        
        print('DONE TESTING')

if __name__ == '__main__':

    abnormality = 'ampulla-of-vater'
    test = test_class(abnormality)
    test.test()  

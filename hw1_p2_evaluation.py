import os
import sys
import argparse
# Import necessary packages.
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from torchvision.datasets import DatasetFolder
from torchvision import models
import mean_iou_evaluate
import imageio

size = (256, 256)
batch_size = 15
transform_0 = [transforms.RandomAffine(degrees=(-10,10), translate=(0.1,0.1), scale=(1,1.5)), transforms.RandomRotation((-10,10)),transforms.RandomHorizontalFlip(p=0.5),transforms.ColorJitter(contrast=(1,1.5), saturation=(1,2)),]
train_tfm = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomChoice(transform_0),                   
    transforms.Resize(size),
    transforms.ToTensor(),
])
test_tfm = transforms.Compose([                  
    transforms.Resize(size),
    transforms.ToTensor(),
])

def readfile(filepath):
    x = []
    file_list = [file for file in os.listdir(filepath) if file.endswith('.jpg')]
    file_list.sort()
    for i, file in enumerate(file_list):
        img = Image.open(os.path.join(filepath, file))
        x.append(test_tfm(img))
        img.close()
    return x, file_list

class P2Dataset(Dataset):
  def __init__(self, X, Y=None, isTrain=False):
    self.data = X
    self.label = Y
    self.isTrain = isTrain
  
  def __getitem__(self, idx):
      if self.label is not None:
        return self.data[idx], self.label[idx]
      else:
        return self.data[idx]

  def __len__(self):
      return len(self.data)

class vgg16_unet(nn.Module):
    def __init__(self, model):
        super(vgg16_unet, self).__init__()
        features = model.features
        self.input_pool2 = nn.Sequential(*list(features.children())[:10])
        self.pool2_pool3 = nn.Sequential(*list(features.children())[10:17])
        self.pool3_pool4 = nn.Sequential(*list(features.children())[17:24])
        self.pool4_pool5 = nn.Sequential(*list(features.children())[24:])

        self.pool5_conv7 = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Dropout2d(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.Dropout2d(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 3, 1, 1),
        )
        self.first_upscore2 = nn.ConvTranspose2d(512, 512, 4, 2, bias=True)
        self.second_upscore2 = nn.ConvTranspose2d(512, 256, 4, 2, bias=True)
        self.third_upscore2 = nn.ConvTranspose2d(256, 128, 4, 2, bias=True)
        self.upscore8 = nn.ConvTranspose2d(512, 128, 16, 8, bias=True)
        self.upscore8_1 = nn.ConvTranspose2d(128, 7, 16, 8, bias=True)
    
    def forward(self, x):
        out1 = self.input_pool2(x) #fcn8: out2 = self.input_pool3(x) unet: out1 = self.input_pool2(x)
        out2 = self.pool2_pool3(out1) #fcn8: no this line unet: need it
        out3 = self.pool3_pool4(out2)
        out4 = self.pool4_pool5(self.pool5_conv7(out3))

        upscorefirst = self.first_upscore2(out4)
        x = out3 + upscorefirst[:, :, 1:upscorefirst.shape[2]-1, 1:upscorefirst.shape[3]-1]
        upscoresecond = self.second_upscore2(x)
        x = out2 + upscoresecond[:, :, 1:upscoresecond.shape[2]-1, 1:upscoresecond.shape[3]-1]
        upscorethird = self.third_upscore2(x)
        x = out1 + upscorethird[:, :, 1:upscorethird.shape[2]-1, 1:upscorethird.shape[3]-1]
        pred = self.upscore8_1(x)
        pred = pred[ :, :, 4:pred.shape[2]-4, 4:pred.shape[3]-4] #for unet
        return pred

if __name__ == '__main__':
    img_path = sys.argv[1]
    outputfile_path = sys.argv[2]
    testing_x, testing_img_name = readfile(img_path)
    testing_set = P2Dataset(testing_x)
    model = vgg16_unet(models.vgg16(pretrained=True))
    testing_loader = DataLoader(testing_set, batch_size=batch_size, shuffle=False)
    model.load_state_dict(torch.load("new_fcn8_best_model_hw1_2.ckpt?dl=1"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.cuda()
    model.eval()

    predictions = []

    for _, batch in enumerate(testing_loader):
        imgs = batch
        with torch.no_grad():
            logits = model(imgs.to(device))
        predictions.extend(logits.argmax(dim=1).cpu().numpy())

    masks_RGB = np.empty((len(predictions), 512, 512, 3))
    for i, p in enumerate(predictions):
      masks_RGB[i, p==0] = [0,255,255]
      masks_RGB[i, p==1] = [255,255,0]
      masks_RGB[i, p==2] = [255,0,255]
      masks_RGB[i, p==3] = [0,255,0]
      masks_RGB[i, p==4] = [0,0,255]
      masks_RGB[i, p==5] = [255,255,255]
      masks_RGB[i, p==6] = [0,0,0]
    masks_RGB = masks_RGB.astype(np.uint8)    
    for i, p in enumerate(masks_RGB):
        imageio.imsave(outputfile_path + testing_img_name[i].split('.')[0] + '.png', p)
    print("finish prediction")  
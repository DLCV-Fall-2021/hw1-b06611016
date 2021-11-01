# -*- coding: utf-8 -*-
"""hw1_p2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1m34fY7UACOZy-Ea8Mojd9OJe70punm0t
"""
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
#from util.wavelet_pool2d import StaticWaveletPool2d, AdaptiveWaveletPool2d
#from util.learnable_wavelets import ProductFilter, SoftOrthogonalWavelet
#from util.pool_select import get_pool
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
'''label_tfm = transforms.Compose([
    transforms.ToTensor(),
])'''
train_path = './hw1_data/p2_data/train/'
val_path = './hw1_data/p2_data/validation/'

def read_masks(filepath):
    '''
    Read masks from directory and tranform to categorical
    '''
    file_list = [file for file in os.listdir(filepath) if file.endswith('.png')]
    file_list.sort()
    n_masks = len(file_list)
    masks = np.empty((n_masks, 512, 512), dtype=np.int_)

    for i, file in enumerate(file_list):
        mask = imageio.imread(os.path.join(filepath, file))
        mask = (mask >= 128).astype(int)
        mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
        masks[i, mask == 3] = 0  # (Cyan: 011) Urban land 
        masks[i, mask == 6] = 1  # (Yellow: 110) Agriculture land 
        masks[i, mask == 5] = 2  # (Purple: 101) Rangeland 
        masks[i, mask == 2] = 3  # (Green: 010) Forest land 
        masks[i, mask == 1] = 4  # (Blue: 001) Water 
        masks[i, mask == 7] = 5  # (White: 111) Barren land 
        masks[i, mask == 0] = 6  # (Black: 000) Unknown 

    return masks

def mean_iou_score(pred, labels):
    '''
    Compute mean IoU score over 6 classes
    '''
    mean_iou = 0
    for i in range(6):
        tp_fp = np.sum(pred == i)
        tp_fn = np.sum(labels == i)
        tp = np.sum((pred == i) * (labels == i))
        iou = tp / (tp_fp + tp_fn - tp)
        mean_iou += iou / 6
        print('class #%d : %1.5f'%(i, iou))
    print('\nmean_iou: %f\n' % mean_iou)

    return mean_iou

def readfile(filepath):
    x = []
    file_list = [file for file in os.listdir(filepath) if file.endswith('.jpg')]
    file_list.sort()
    for i, file in enumerate(file_list):
        #print(file)
        img = Image.open(os.path.join(filepath, file))
        x.append(test_tfm(img))
        img.close()
    y = read_masks(filepath)
    return x, y

class P2Dataset(Dataset):
  def __init__(self, X, Y, isTrain):
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
        # self.upscore2 = nn.ConvTranspose2d(512, 256, 4, 2, bias=False)
        # self.upscore4 = nn.ConvTranspose2d(512, 256, 8, 4, bias=False)
        self.upscore8 = nn.ConvTranspose2d(512, 128, 16, 8, bias=True)
        self.upscore8_1 = nn.ConvTranspose2d(128, 7, 16, 8, bias=True)
        # self.upscore16 = nn.ConvTranspose2d(256, 7, 32, 16, bias=False)
    
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
        #upscore8 = self.upscore8(out4)
        #upscore4 = self.upscore4(out3)
        #upscore2 = self.upscore2(out2)
        #upscore2 = self.upscore2(out2)
        # upscore4 = self.upscore4(out4) #fcn8
        # upscore2 = self.upscore2(out3) #fcn8
        # combine_fig = out2 + upscore2[ :, :, 1:upscore2.shape[2]-1, 1:upscore2.shape[3]-1] + upscore4[ :, :, 2:upscore4.shape[2]-2, 2:upscore4.shape[3]-2] #fcn8
        #print(upscore2.shape)
        #print(upscore4.shape)
        #print(upscore8.shape)
        #combine_fig = out1 + upscore2[ :, :, 1:upscore2.shape[2]-1, 1:upscore2.shape[3]-1] + upscore4[ :, :, 2:upscore4.shape[2]-2, 2:upscore4.shape[3]-2] + upscore8[:, :, 4:upscore8.shape[2]-4, 4:upscore8.shape[3]-4]
        #print(combine_fig.shape)
        #pred = self.upscore8_1(combine_fig)
        #print(pred.shape)
        #pred = self.upscore16(combine_fig) #fcn8
        pred = pred[ :, :, 4:pred.shape[2]-4, 4:pred.shape[3]-4] #for unet
        #pred = pred[ :, :, 8:pred.shape[2]-8, 8:pred.shape[3]-8] #fcn8
        return pred

train_x, train_y = readfile(train_path)
val_x, val_y = readfile(val_path)
train_set = P2Dataset(train_x, train_y, True)
val_set = P2Dataset(val_x, val_y, False)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
model = vgg16_unet(models.vgg16(pretrained=True))
print(model)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model = model.cuda()
criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.Adam(model.parameters(), lr=0.0005,  betas=(0.9, 0.99))
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
n_epochs = 2500
best_miou = 0.0
best_model_path = '../drive/MyDrive/DLCV/hw1/new_fcn8_best_model_hw1_2.ckpt' #fcn_best_model_hw1_2: 0.68417 #transfer fcn8 to unet
model_path = '../drive/MyDrive/DLCV/hw1/new_fcn8_model_hw1_2.ckpt'     #unet:  0.71293; epoch:  260; path: ../drive/MyDrive/DLCV/hw1/new_fcn8_best_model_hw1_2.ckpt #new path: ../drive/MyDrive/DLCV/hw1/fcn8_best_model_hw1_2.ckpt miou: 0.70900
optimizer_path = '../drive/MyDrive/DLCV/hw1/new_fcn8_optimizer_hw1_2.ckpt'

for epoch in range(n_epochs):
    # ---------- Training ----------
    # Make sure the model is in train mode before training.
    model.train()

    # These are used to record information in training.
    train_loss = []
    train_preds = []
    train_miou = 0.0
    train_label = []

    # Iterate the training set by batches.
    for _,batch in enumerate(train_loader):
        #print("here")
        # A batch consists of image data and corresponding labels.
        imgs, labels = batch

        # Forward the data. (Make sure data and model are on the same device.)
        logits = model(imgs.to(device))

        # Calculate the cross-entropy loss.
        # We don't need to apply softmax before computing cross-entropy as it is done automatically.
        #labels = torch.tensor(labels, dtype=torch.long)
        loss = criterion(logits, labels.to(device))

        # Gradients stored in the parameters in the previous step should be cleared out first.
        optimizer.zero_grad()

        # Compute the gradients for parameters.
        loss.backward()

        # Clip the gradient norms for stable training.
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

        # Update the parameters with computed gradients.
        optimizer.step()

        # Compute the accuracy for current batch.
        #acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
        #print(logits.shape)
        logits = logits.argmax(dim=1)
        #print(logits.shape)
        # Record the loss and accuracy.
        train_loss.append(loss.item())
        train_preds.extend(logits.cpu().detach().numpy())
        train_label.extend(labels.cpu().detach().numpy())
        #print(len(train_preds))
    # The average loss and accuracy of the training set is the average of the recorded values.
    train_loss = sum(train_loss) / len(train_loss)
    #train_acc = sum(train_accs) / len(train_accs)
    train_preds = np.array(train_preds, dtype=np.int_)
    train_label = np.array(train_label, dtype=np.int_)
    #print(train_preds)
    train_miou = mean_iou_score(train_preds, train_label)

    # Print the information.
    print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, miou = {train_miou:.5f}")

    # ---------- Validation ----------
    # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
    model.eval()

    # These are used to record information in validation.
    valid_loss = []
    valid_preds = []
    valid_miou = 0.0
    # Iterate the validation set by batches.
    for _,batch in enumerate(valid_loader):

        # A batch consists of image data and corresponding labels.
        imgs, labels = batch

        # We don't need gradient in validation.
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
          logits = model(imgs.to(device))

        # We can still compute the loss (but not the gradient).
        #labels = torch.tensor(labels, dtype=torch.long)
        loss = criterion(logits, labels.to(device))

        # Compute the accuracy for current batch.
        #acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
        logits = logits.argmax(dim=1)
        # Record the loss and accuracy.
        valid_loss.append(loss.item())
        valid_preds.extend(logits.cpu().detach().numpy())

    # The average loss and accuracy for entire validation set is the average of the recorded values.
    valid_loss = sum(valid_loss) / len(valid_loss)
    #valid_acc = sum(valid_accs) / len(valid_accs)
    valid_preds = np.array(valid_preds, dtype=np.int_)
    valid_miou = mean_iou_score(valid_preds, val_y)
    if valid_miou > best_miou:
      best_miou = valid_miou
      torch.save(model.state_dict(), best_model_path)
      print('saving model with acc {:.3f}'.format(best_miou))
    torch.save(optimizer.state_dict(), optimizer_path)
    torch.save(model.state_dict(), model_path)  
    # Print the information.
    print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, miou = {valid_miou:.5f}")
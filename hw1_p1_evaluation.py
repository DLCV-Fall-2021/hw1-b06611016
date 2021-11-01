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

batch_size = 50
size = (224, 224)
#transforms.RandomAffine(degrees=(-10,10), translate=(0.1,0.1), scale=(1,1.5)),
transform_0 = [transforms.RandomRotation((-10,10)),transforms.RandomHorizontalFlip(p=0.5),transforms.ColorJitter(contrast=(1,1.5), saturation=(1,2)),]
transform_1 = [transforms.ColorJitter(contrast=(1,1.5), saturation=(1,2)),]
transform_2 = [transforms.RandomAffine(degrees=(-10,10), translate=(0.1,0.1), scale=(1,1.5)),transforms.ColorJitter(contrast=(1,1.5), saturation=(1,2)),transforms.RandomHorizontalFlip(p=0.5),]
train_tfm = transforms.Compose([
    transforms.RandomChoice(transform_0),
    transforms.Resize(size),
    transforms.ToTensor(),
])

train_tfm_1 = transforms.Compose([
    transforms.RandomChoice(transform_1),
    transforms.Resize(size),
    transforms.ToTensor(),
])
train_tfm_2 = transforms.Compose([
    transforms.RandomChoice(transform_2),
    transforms.Resize(size),
    transforms.ToTensor(),
])
# We don't need augmentations in testing and validation.
# All we need here is to resize the PIL image and transform it into Tensor.
test_tfm = transforms.Compose([
    transforms.Resize(size),
    transforms.ToTensor(),
])

def readfile(filepath):
    x = []
    file_list = [file for file in os.listdir(filepath) if file.endswith('.png')]
    file_list.sort()
    for i, file in enumerate(file_list):
        img = Image.open(os.path.join(filepath, file))
        x.append(test_tfm(img))
        img.close()
    return x, file_list

class P1Dataset(Dataset):
  def __init__(self, X, Y=None, isTrain=False):
    self.data = X
    self.label = Y
    self.isTrain = isTrain
  
  def __getitem__(self, idx):
      if self.label is not None:
        if self.isTrain:
          return train_tfm(self.data[idx]), self.label[idx]
        else:
          return test_tfm(self.data[idx]), self.label[idx]
      else:
          return self.data[idx]

  def __len__(self):
      return len(self.data)

if __name__ == '__main__':
    img_path = sys.argv[1]
    outputfile_path = sys.argv[2]
    testing_x, testing_img_name = readfile(img_path)
    testing_set = P1Dataset(testing_x)
    model = models.resnet101(pretrained=True)
    testing_loader = DataLoader(testing_set, batch_size=batch_size, shuffle=False)
    fc_inputs = model.fc.in_features
    model.fc = nn.Sequential(
        #nn.BatchNorm1d(fc_inputs),
        nn.Linear(fc_inputs, 50),
        #nn.ReLU()
    )
    model.load_state_dict(torch.load("new_best_model_hw1_1.ckpt?dl=1"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.cuda()
    model.eval()

    predictions = []

    for _, batch in enumerate(testing_loader):
        imgs = batch
        with torch.no_grad():
            logits = model(imgs.to(device))
        predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
    
    with open(outputfile_path, "w") as f:
        f.write("image_id,label\n")
        for i, pred in enumerate(predictions):
            f.write(f"{testing_img_name[i]}, {pred}\n")
    print("finish prediction")
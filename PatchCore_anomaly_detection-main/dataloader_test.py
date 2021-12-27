import os
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import argparse
import torch
from torch.nn import functional as F
from torch import nn
from PIL import Image
import glob
from mydataset import MyDataset

def main():
    #imagenet
    train_dataset_path = '/youtu-public/YOUTU_FX3/AppleML/50k_production'
    mean_train = [0.485, 0.456, 0.406]
    std_train = [0.229, 0.224, 0.225]

    data_transforms = transforms.Compose([
                    transforms.Resize((328, 328), Image.ANTIALIAS),
                    transforms.ToTensor(),
                    transforms.CenterCrop(224),
                    transforms.Normalize(mean=mean_train,
                                        std=std_train)])
    gt_transforms = transforms.Compose([
                    transforms.Resize((328, 328)),
                    transforms.ToTensor(),
                    transforms.CenterCrop(224)])
    image_datasets = MyDataset(root=train_dataset_path, transform=data_transforms, gt_transform=gt_transforms)
    train_loader = DataLoader(image_datasets, batch_size=8, shuffle=True, num_workers=1) #, pin_memory=True)


    for (idx,data)in train_loader:
        print(idx)
        x, _, file_name, _ = data
    
if __name__ == '__main__':
    main()
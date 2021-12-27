import os
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import argparse
import torch
from torch.nn import functional as F
from torch import nn
from PIL import Image
import glob

class MyDataset(Dataset):
    def __init__(self, root, transform, gt_transform):

        self.img_path = root
        # else:
        #     self.img_path = os.path.join(root, 'test')
        #     self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset() # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []
        #gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)
        
        for defect_type in defect_types:
            if defect_type == 'imgs_folder':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.jpg")
                img_tot_paths.extend(img_paths)
                #gt_tot_paths.extend([0]*len(img_paths))
                tot_labels.extend([0]*len(img_paths))
                tot_types.extend(['good']*len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.jpg")
                #gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths.sort()
                #gt_paths.sort()
                img_tot_paths.extend(img_paths)
                #gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1]*len(img_paths))
                tot_types.extend([defect_type]*len(img_paths))

        #assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"
        
        return img_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, label, img_type = self.img_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        return img, label, os.path.basename(img_path[:-4]), img_type



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


class traindataset(Dataset):
    def __init__(self, root, transform, gt_transform):
        self.label_path = os.path.join(root, 'ok_list.txt')
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        # self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset() # self.labels => good : 0, anomaly : 1
        self.img_paths, self.labels, self.types = self.load_dataset() # self.labels => good : 0, anomaly : 1


    def load_dataset(self):
        img_tot_paths = []
        tot_labels = []
        tot_types = []
        with open(self.label_path, 'r') as f:
            for line in f:
                line = line.strip()
                temp_path = line
                img_tot_paths.append(temp_path)
                temp_label = 0
                tot_labels.append(int(temp_label))
                tot_types.append('good' if temp_label=='0' else 'defect')
                del temp_path
        return img_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, label, img_type = self.img_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        return img, label, os.path.basename(img_path[:-4]), img_type
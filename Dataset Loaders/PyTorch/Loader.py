from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import pandas as pd
import numpy as np
import os
from PIL import Image
import torch
from Sampler import ImbalancedDatasetSampler
import torch.utils.data.sampler


class DeepWeeds(Dataset):
    def __init__(self, image_path, csv_path, transform=None):
        self.image_path = image_path
        self.to_tensor = transforms.ToTensor()
        self.data_labels = pd.read_csv(csv_path)
        self.labels = np.asarray(self.data_labels.iloc[:, 0])
        self.image_arr = np.asarray(self.data_labels.iloc[:, 1])
        self.data_len = len(self.data_labels.index)
        self.transform = transform

    def __getitem__(self, index):
        self.single_image_label = self.image_arr[index]
        label = self.single_image_label
        img_as_img = Image.open(os.path.join(self.image_path, self.labels[index]).replace("\\","/"))
        if self.transform is not None:
            img_as_ten = self.transform(img_as_img)
        else:
            img_as_ten = self.to_tensor(img_as_img)
        return (img_as_ten, label)

    def __len__(self):
        return self.data_len


def loadDeepWeeds(batch_size=64, shuffle=True, num_workers=2, pre_processing_transform=None, use_imbalanced_dataset_sampler=True, image_directory_path=None, train_csv_path=None, test_csv_path=None):
    train_dataset = DeepWeeds(image_directory_path, csv_path=train_csv_path, transform=pre_processing_transform)
    test_dataset = DeepWeeds(image_directory_path, csv_path=test_csv_path, transform=pre_processing_transform)
    if use_imbalanced_dataset_sampler:
        train_data_loader = torch.utils.data.DataLoader(train_dataset, sampler=ImbalancedDatasetSampler(train_dataset), batch_size=batch_size, num_workers=num_workers)
        test_data_loader = torch.utils.data.DataLoader(test_dataset, sampler=ImbalancedDatasetSampler(test_dataset), batch_size=int(batch_size/2), num_workers=num_workers)
    else:
        train_data_loader = torch.utils.data.DataLoader(train_dataset, sampler=SubsetRandomSampler(train_dataset), batch_size=batch_size, num_workers=num_workers)
        test_data_loader = torch.utils.data.DataLoader(test_dataset, sampler=SubsetRandomSampler(test_dataset), batch_size=int(batch_size/2), num_workers=num_workers)
    return train_data_loader, test_data_loader

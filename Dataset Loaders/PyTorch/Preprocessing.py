from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler


# Image Preprocessing Techniques
def IPT():
    data_transforms = transforms.Compose([
                transforms.RandomResizedCrop(size=224),
                transforms.ToTensor()
    ])
    return data_transforms

# Further Image Preprocessing Techniques
def FIPT():
    data_transforms = transforms.Compose([
                transforms.RandomResizedCrop(size=224),
                transforms.RandomAffine(15),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return data_transforms

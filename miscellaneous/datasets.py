import glob
import os

# Ignore warnings
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

import miscellaneous.utilities as utilities

warnings.filterwarnings("ignore")

class ImageDataset(Dataset):
    """
    Implements a Dataset subclass that reads .png images from a folder of data
    (where each element is assumed to be a .png representing gray-scale image) 
    and converts it to either a numpy array or a pytorch Tensor.

    Arguments:
        data_path: str, (Relative) path to the dataset.
        numpy: bool, if True, returns a numpy array, a pytorch Tensor is returned  
                    otherwise. Default: False.
    """
    def __init__(self, data_path, numpy=False, label_path=None):
        self.data_path = data_path

        self.label_path = label_path
        self.img_name_list = sorted(glob.glob(os.path.join(self.data_path, '*.png')))
        if self.label_path is not None:
            self.label_name_list = sorted(glob.glob(os.path.join(self.label_path, '*.png')))

        self.numpy = numpy

    def __len__(self):
        return len(self.img_name_list)
    
    def __getitem__(self, index):
        img = Image.open(self.img_name_list[index])
        img = utilities.normalize(np.expand_dims(np.array(img)[:, :, 0], 0))

        if self.label_path is not None:
            label = Image.open(self.label_name_list[index])
            label = utilities.normalize(np.expand_dims(np.array(label)[:, :, 0], 0))

            if not self.numpy:
                label = torch.Tensor(label)

        if not self.numpy:
            img = torch.Tensor(img)
        
        if self.label_path is not None:
            return img, label
        return img

    def get_name(self, index):
        return self.img_name_list[index].split("\\")[-1]

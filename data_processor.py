import torch
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F

class Nas_Data(Dataset):
    def __init__(self,x,y=None,test=False):
        self.image=x
        self.label=y
        self.test=test
    def __len__(self):
        return len(self.image)
    def __getitem__(self,item):
        img = self.image[item]
        h, w = img.shape
        if height %2 != 0:
            F.pad(input = img, pad = (0, 0, 0, 1), mode='constant', value=0)
        if width % 2 == 0:
            F.pad(input = img, pad = (0, 0, 1, 0), mode='constant', value=0)
        if self.test != True:
            return torch.from_numpy(self.image[item]).float(),torch.tensor(self.label[item])
        else:
            return torch.from_numpy(self.image[item]).float()

class DataProcessor:
    """
    -===================================================================================================================
    INIT ===============================================================================================================
    ====================================================================================================================
    The DataProcessor class will receive the following inputs:
        * train_x: numpy array of shape [n_train_datapoints, channels, height, width], these are the training inputs
        * train_y: numpy array of shape [n_train_datapoints], these are the training labels
        * valid_x: numpy array of shape [n_valid_datapoints, channels, height, width], these are the validation inputs
        * valid_y: numpy array of shape [n_valid_datapoints], these are the validation labels
        * test_x: numpy array of shape [n_valid_datapoints, channels, height, width], these are the test inputs
        * metadata: A dictionary with information about this dataset, with the following keys:
            'num_classes' : The number of output classes in the classification problem
            'codename' : A unique string that represents this dataset
            'input_shape': A tuple describing [n_total_datapoints, channel, height, width] of the input data
            'time_remaining': The amount of compute time left for your submission

    You can modify or add anything into the metadata that you wish, if you want to pass messages between your classes

    """

    def __init__(self, train_x, train_y, valid_x, valid_y, test_x, metadata):
        self.train=Nas_Data(train_x,train_y)
        self.valid=Nas_Data(valid_x,valid_y)
        self.test=Nas_Data(test_x,test=True)
        self.metadata=metadata


    """
    ====================================================================================================================
    PROCESS ============================================================================================================
    ====================================================================================================================
    This function will be called, and it expects you to return three outputs:
        * train_loader: A Pytorch dataloader of (input, label) tuples
        * valid_loader: A Pytorch dataloader of (input, label) tuples
        * test_loader: A Pytorch dataloader of (inputs)  <- Make sure shuffle=False and drop_last=False!

    See https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader for more info.  

    Here, you can do whatever you want to the input data to process it for your NAS algorithm and training functions
    """

    def process(self):
        train_loader=torch.utils.data.DataLoader(self.train,batch_size=32,shuffle=True,drop_last=True)
        valid_loader=torch.utils.data.DataLoader(self.valid,batch_size=32,shuffle=True)
        test_loader=torch.utils.data.DataLoader(self.test,batch_size=32,shuffle=False,drop_last=False)

        return train_loader, valid_loader, test_loader

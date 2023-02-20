import os 
import torch
from PIL import Image 
import numpy as np

class Dataset:
    #get folder names from data set
    def __init__(self, root_dir, transform):
        root_dir = 'Soil types'
        assert os.path.exists(root_dir), 'Data does not exist'

        data_files = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir)]
        data_label = 0
        self.data_all={}
        self.data_all['label'] = []
        self.data_all['file_name'] = []

        for file in data_files:    
            data_samples = os.listdir(file)
            for data in data_samples:
                data_file_name = os.path.join(file, data)
                self.data_all['label'].append(data_label)
                self.data_all['file_name'].append(data_file_name)
            data_label += 1 

    #check data set length 
    def __len__(self):
        return len(self.data_all['label'])

    #get an item from dataset 
    def __getitem__(self, index):
        # print(self.data[index])
        image = Image.open(self.data_all['file_name'][index])
        newsize = (256,256)
        image = image.resize(newsize)
        image = np.array(image)
        label = np.array(self.data_all['label'][index])


        if self.transform:
            image = torch.from_numpy(image)
            label = torch.from_numpy(label)

        data_point = {'label': label, 'image': image}
        return data_point
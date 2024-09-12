import torch 
from torch.utils.data import Dataset, DataLoader
import numpy as np 
import os 
from torch import sqrt 
from tqdm import tqdm
import pandas as pd

BASE_PATH = '/home/winston/Documents/CT_images_pre_training_176'
TEST_BASE_PATH= None

class CT176Data(Dataset):
    def __init__(self, base_path = None, mode = 'train',  transform=None, use_z_score=False, pre_load = False):
        
        self.transform = transform
        self.use_z_score = use_z_score
        self.data_range = 255
        # self.return_original = return_original
        if base_path is None or base_path == "":
            self.base_path = BASE_PATH if mode != 'test' else TEST_BASE_PATH
        else:
            self.base_path = base_path
        print('base path', self.base_path)
        self.mode = mode
        
        # for modality in ['flair', 't1', 't1ce', 't2']:
        #     file_path = os.path.join(base_path, modality)
            
        #     data_raw = np.load(file_path)
        # setattr(self, f'{modality}_data', data_raw)
        self.pre_load = pre_load
        if self.pre_load:
            self.load_data(self.base_path)
        else:
            # self.t1_base_path = os.path.join(self.base_path, "t1")
            self.data_path_list = sorted(os.listdir(self.base_path))
        
        if not self.use_z_score:
            print("data normalization is done using min-max scaling")
        if self.mode == "test":
            pass 

        
        
    def load_data(self, base_path):
        self.data = []
        self.label = []
        
        for file in tqdm(os.listdir(base_path)):
            if file.endswith('.npy'):
            
                volume = np.load(os.path.join(base_path, file)) / self.data_range
                
                self.data.append(volume)

    
    def __len__(self):
        if self.pre_load:
            return len(self.data)
        else:
            return len(self.data_path_list)
    def __getitem__(self, idx):
        return_item = []
        if self.pre_load:
            volume = torch.tensor(self.data[idx], dtype = torch.float)
        else:
            volume = torch.tensor(np.load(os.path.join(self.base_path, self.data_path_list[idx])), dtype = torch.float) / self.data_range
        
        volume = self._normalize_data(volume)
        volume = volume.unsqueeze(0)
        if self.transform: 
            volume = self.transform(volume)
        return_item.append(volume)
        if self.mode == "test":
            pass
        else:
            return_item.append(0)
        return return_item
        
        
    def _normalize_data(self, volume):
        if self.use_z_score:
            # Since this is a single channel image so, we can ignore the `axis` parameter
            return (volume - volume.mean()) / sqrt(volume.var())
        max_val, min_val = volume.max(), volume.min()
        volume = (volume - min_val) / (max_val - min_val)
        return volume
    
if __name__ == "__main__":
    base_path = '/home/winston/Documents/CT_images_pre_training_176'
    dataset = CT176Data(base_path, mode = 'train')
    print(len(dataset))
    print(dataset[0][0].shape)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    for sample, _ in dataloader:
        # print(data.shape, seg.shape)
        print(sample.shape)
        # print(batch[0].max(), batch[0].min())
        break

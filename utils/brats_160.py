import torch 
from torch.utils.data import Dataset, DataLoader
import numpy as np 
import os 
from torch import sqrt 
from tqdm import tqdm
import pandas as pd
import configparser

LABLE_TYPE = ['IDH', '1p19q']

def get_config(section, name = 'BASE_PATH'):
    current_path = os.path.dirname(os.path.dirname(__file__))
    print(current_path)
    config = configparser.ConfigParser()
    # config.read(config_file)
    config.read(os.path.join(current_path, 'config.ini'), encoding='utf-8')
    
    return config[section][name]

class Brats160Data(Dataset):
    def __init__(self, base_path = None, mode = 'train', label_type = "IDH",  transform=None, use_z_score=False, pre_load = False):
        # self.data = np.load(filename)
        # self.label = np.load(label_name) if label_name else None
        
        self.transform = transform
        self.use_z_score = use_z_score
        self.data_range = 255
        assert label_type in LABLE_TYPE
        self.label_type = label_type
        # self.return_original = return_original
        if base_path is None or base_path == "":
            self.base_path = get_config('brats_160', name='BASE_PATH') if mode != 'test' else get_config('brats_160', name='TEST_BASE_PATH')
        else:
            self.base_path = base_path
        print('base path', self.base_path)
        self.mode = mode
        

        self.pre_load = pre_load
        if self.pre_load:
            self.load_data(self.base_path)
        else:
            self.t1_base_path = os.path.join(self.base_path, "t1")
            self.t1_path_list = sorted(os.listdir(os.path.join(self.base_path, 't1')))
        
        if not self.use_z_score:
            print("data normalization is done using min-max scaling")
        if self.mode == "test":
            self.df_labels = pd.read_excel(os.path.join(self.base_path, 'Genetic_and_Histological_labels.xlsx'))
            self.df_labels['IDH'] = self.df_labels['IDH'].replace(-1, 2)
            self.df_labels['1p19q'] = self.df_labels['1p19q'].replace(-1, 2)
            no_valid_list = []
            for i, t1 in enumerate( self.t1_path_list):
                name = t1.split('.')[0].replace('_t1', '')
                label = self.df_labels.loc[self.df_labels['Subject'] == name].values[0][1]
                if self.label_type == LABLE_TYPE[1]:
                    label = self.df_labels.loc[self.df_labels['Subject'] == name].values[0][2]
                if label == 2:
                    no_valid_list.append(i)
            self.t1_path_list = [i for j, i in enumerate(self.t1_path_list) if j not in no_valid_list]

        
    def load_data(self, base_path):
        self.data = []
        self.label = []
        middle_path = os.path.join(base_path, 't1')
        for file in tqdm(os.listdir(middle_path)):
            if file.endswith('.npy'):
                # print(file)
                file_name = file.split('.')[0]
                t1 = np.load(os.path.join(middle_path, file)) / self.data_range
                t2 = np.load(os.path.join(base_path, 't2', file_name.replace('t1', 't2') + ".npy"),) /self.data_range
                t1ce = np.load(os.path.join(base_path, 't1ce', file_name.replace('t1', 't1ce') + ".npy"), ) / self.data_range
                flair = np.load(os.path.join(base_path, 'flair', file_name.replace('t1', 'flair') + ".npy")) / self.data_range
                seg = np.load(os.path.join(base_path, 'seg_mask', file_name.replace('t1', 'seg') + ".npy")) / self.data_range
                
                volume = np.stack([t1, t2, t1ce, flair], axis=0)
                self.data.append(volume)
                self.label.append(seg)
    
    def __len__(self):
        if self.pre_load:
            return len(self.data)
        else:
            return len(self.t1_path_list)
    
    def __getitem__(self, idx):
        return_item = []
        if self.pre_load:
            volume = torch.tensor(self.data[idx], dtype = torch.float)
            seg = torch.tensor(self.label[idx], dtype = torch.float)
        else:
            t1 = np.load(os.path.join(self.t1_base_path, self.t1_path_list[idx])) / self.data_range
            t2 = np.load(os.path.join(self.base_path, 't2', self.t1_path_list[idx].replace('t1', 't2'))) / self.data_range
            t1ce = np.load(os.path.join(self.base_path, 't1ce', self.t1_path_list[idx].replace('t1', 't1ce'))) / self.data_range
            flair = np.load(os.path.join(self.base_path, 'flair', self.t1_path_list[idx].replace('t1', 'flair'))) / self.data_range
            
            volume = np.stack([t1, t2, t1ce, flair], axis=0)
            volume = torch.tensor(volume, dtype = torch.float)
            try:
                seg = np.load(os.path.join(self.base_path, 'seg_mask', self.t1_path_list[idx].replace('t1', 'seg'))) / self.data_range
                seg = torch.tensor(seg, dtype = torch.float)
            except:
                seg = torch.tensor(0, dtype = torch.float)
        
        volume = self._normalize_data(volume)
        if self.transform:
            volume = self.transform(volume)
        return_item.append(volume)
        if self.mode == "test":
            name = self.t1_path_list[idx].split('.')[0].replace('_t1', '')
            # print(name)
            # if self.label_type == LABLE_TYPE[0]:
            label = self.df_labels.loc[self.df_labels['Subject'] == name].values[0][1]
            if self.label_type == LABLE_TYPE[1]:
                label = self.df_labels.loc[self.df_labels['Subject'] == name].values[0][2]
            # print(label)
            label = np.array(label)
            return_item.append(torch.from_numpy(label).long())
        else:
            return_item.append(seg)
        return return_item
        
        
    def _normalize_data(self, volume):
        if self.use_z_score:
            # Since this is a single channel image so, we can ignore the `axis` parameter
            return (volume - volume.mean()) / sqrt(volume.var())
        max_val, min_val = volume.max(), volume.min()
        volume = (volume - min_val) / (max_val - min_val)
        return volume
    
if __name__ == "__main__":
    import torchio as tio
    transforms_ = [
        tio.RandomAffine(),
        tio.RandomNoise(std=0.1),
        tio.RandomGamma(log_gamma=(-0.3, 0.3)),
        tio.RandomAnisotropy(axes = (0, 1), downsampling=(2, 4)),
        tio.RandomMotion(degrees = (0, 45), translation=(1, 3)),
        ]
        
        
    train_transforms1 = tio.Compose(transforms_)
    base_path = ''
    dataset = Brats160Data(base_path, mode = 'test', transform=train_transforms1, label_type='1p19q')
    print(len(dataset))
    print(dataset[0][0].shape, dataset[0][1].shape)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    # data_list = [data[1] for data in dataset]
    # print(data_list)
    for batch in dataloader:
        print(batch[1])
        # print(data.shape, seg.shape)
        # print(batch[0].shape,)
        # break

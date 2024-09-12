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


class Uni160Data(Dataset):
    def __init__(self, base_path = None, mode = 'train', label_type = "IDH",  transform=None, use_z_score=False):
        # self.data = np.load(filename)
        # self.label = np.load(label_name) if label_name else None
        
        self.transform = transform
        self.use_z_score = use_z_score
        self.data_range = 255
        assert label_type in LABLE_TYPE
        self.label_type = label_type
        # self.return_original = return_original
        if base_path is None or base_path == "":
            self.base_path = get_config("uni_160") 
        else:
            self.base_path = base_path
        print('base path', self.base_path)
        self.mode = mode

        self.label = []
        self.t1_path_list = []
        
        all_sub_folder = os.listdir(self.base_path)
        all_sub_folder = [x for x in all_sub_folder if os.path.isdir(os.path.join(self.base_path, x))]
        
        label_file = pd.read_csv(os.path.join(self.base_path, 'phenoData.csv'))
        if mode == 'train':
            label_file = label_file[(pd.isna(label_file['IDH'])) & 
                                    (pd.isna(label_file['CoDel1p19q']))]
        elif mode == 'test':
            self.label_type = label_type
            label_name = 'IDH' if label_type == '1p19q' else 'CoDel1p19q'
            label_file = label_file[(~pd.isna(label_file[label_name])) & (label_file['Dataset'].isin( all_sub_folder))]
            self.label = label_file[label_name].values
            
        
        patient_list = label_file['Patient'].values
        for patient in patient_list:
            folder = label_file.loc[label_file['Patient'] == patient]['Dataset'].values[0]
            # print(folder)
            if folder in all_sub_folder:
                t1_path = os.path.join(self.base_path, folder, 't1', patient + '_t1.npy')
                self.t1_path_list.append(t1_path)
        if mode == "test":
            assert len(self.label) == len(self.t1_path_list)
        
    def __len__(self):
        
        return len(self.t1_path_list)
    def __getitem__(self, idx):
        return_item = []
        t1_path = self.t1_path_list[idx]
        
        t1 = np.load(t1_path) / self.data_range
        t2 = np.load(t1_path.replace('t1', 't2')) / self.data_range
        t1ce = np.load(t1_path.replace('t1', 't1ce')) / self.data_range
        flair = np.load(t1_path.replace('t1', 'flair')) / self.data_range
        
        volume = np.stack([t1, t2, t1ce, flair], axis=0)
        volume = torch.tensor(volume, dtype = torch.float)

        
        volume = self._normalize_data(volume)
        if self.transform:
            volume = self.transform(volume)
        return_item.append(volume)
        if self.mode == "test":
            label = self.label[idx]
            return_item.append(np.array(label))
        else:
            return_item.append(np.array(-1))
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
    base_path = ''
    transforms_ = [
        tio.RandomAffine(),
        tio.RandomNoise(std=0.1),
        tio.RandomGamma(log_gamma=(-0.3, 0.3)),
        tio.RandomAnisotropy(axes = (0, 1), downsampling=(2, 4)),
        tio.RandomMotion(degrees = (0, 45), translation=(1, 3)),
        ]
        
        
    train_transforms1 = tio.Compose(transforms_)
    dataset = Uni160Data(base_path, mode = 'train', transform=train_transforms1)
    print(len(dataset))
    print(dataset[0][0].shape, dataset[0][1].shape)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    for batch in dataloader:
        # print(data.shape, seg.shape)
        # print(batch[0].shape, batch[1].shape)
        pass 

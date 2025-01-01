import numpy as np
import scipy.io
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import torch
import h5py

class GetBBCSport(Dataset):
    def __init__(self, name):
        data = h5py.File('./data/' + name + '.mat')
        self.V1 = np.array(data[data['X'][0,0]].astype(np.float32)).T
        self.V2 = np.array(data[data['X'][0,1]].astype(np.float32)).T
        self.Y = np.array(data['Y'].astype(np.int32)).T
    def __len__(self):
        return 544

    def __getitem__(self, idx):
        x1 = self.V1[idx].reshape(3183)
        x2 = self.V2[idx].reshape(3203)
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()
def load_data(data_name):
    if data_name == "BBCSport":
        dataset = GetBBCSport(data_name)
        dims = [3183, 3203]
        view = 2
        data_size = 544
        class_num = 5
    else:
        raise NotImplementedError
    return dataset, dims, view, data_size, class_num

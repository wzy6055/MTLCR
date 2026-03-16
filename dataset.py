import os
from torch.utils.data import Dataset
import numpy as np

import skimage.io as io
from pathlib import Path

class CUHKCRDataset(Dataset):
    def __init__(self, data_dir, latent_dir=None, mode='train', val_size=None):
        super(CUHKCRDataset, self).__init__()
        self.data_dir = data_dir
        self.mode = mode
        if self.mode == 'train':
            img_list = np.loadtxt(os.path.join(data_dir, 'train.txt'), dtype=str)
        else:
            img_list = np.loadtxt(os.path.join(data_dir, 'test.txt'), dtype=str)
            if val_size is not None:
                img_list = img_list[:val_size]

        self.clear_list = [os.path.join(self.data_dir, 'clear', filename) for filename in img_list]
        self.cloudy_list = [os.path.join(self.data_dir, 'cloudy', filename) for filename in img_list]

        if latent_dir is not None:
            self.latent=True
            self.clear_latent_list = [(Path(latent_dir) / 'clear' / Path(file_path).stem).with_suffix('.npy') for file_path in self.clear_list]
            self.cloudy_latent_list = [(Path(latent_dir) / 'cloudy' / Path(file_path).stem).with_suffix('.npy') for file_path in self.cloudy_list]
        else:
            self.latent=False

    def __len__(self):
        return len(self.clear_list)

    def __getitem__(self, idx):

        clear = io.imread(self.clear_list[idx]).astype(np.float32)
        cloudy = io.imread(self.cloudy_list[idx]).astype(np.float32)

        clear = (clear / 255) * 2 - 1
        cloudy = (cloudy / 255) * 2 - 1

        clear = clear.transpose(2, 0, 1)
        cloudy = cloudy.transpose(2, 0, 1)

        if self.latent:
            cloudy_latent = np.load(self.cloudy_latent_list[idx])
            clear_latent = np.load(self.clear_latent_list[idx])
            data = {'raw_clear': clear,
                    'raw_cloudy': cloudy,
                    # 'cond': cloudy,
                    'cloudy_latent': cloudy_latent,
                    'clear_latent': clear_latent,
                    'filename': self.clear_list[idx]
                    }
        else:
            data = {'clear': clear,
                    'cloudy': cloudy,
                    'cond': cloudy,
                    'filename': self.clear_list[idx]
                    }
        return data

class ISPRSDataset(Dataset):
    def __init__(self, cfg, mode='train', val_size=None):
        super(ISPRSDataset, self).__init__()
        assert cfg.cloud_type in ['thin', 'thick']

        self.data_dir = cfg.data_dir
        self.mode = mode
        self.cloud_type = cfg.cloud_type

        if self.mode == 'train':
            img_list = np.loadtxt(os.path.join(self.data_dir, 'train.txt'), dtype=str)
        else:
            img_list = np.loadtxt(os.path.join(self.data_dir, 'test.txt'), dtype=str)
            if val_size is not None:
                img_list = img_list[:val_size]

        self.img_list = img_list

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        clear = io.imread(os.path.join(self.data_dir, 'clear', self.img_list[idx])).astype(np.float32)[:, :, :3]
        cloudy = io.imread(os.path.join(self.data_dir, self.cloud_type, self.img_list[idx].replace('tif', 'png'))).astype(np.float32)[:, :, :3]

        clear = (clear / 255) * 2 - 1
        cloudy = (cloudy / 255) * 2 - 1

        clear = clear.transpose(2, 0, 1)
        cloudy = cloudy.transpose(2, 0, 1)

        data = {'clear': clear,
                'cloudy': cloudy,
                'cond': cloudy,
                'filename': self.img_list[idx]
                }
        return data


if __name__ == '__main__':
    from types import SimpleNamespace
    from torch.utils import data

    # dataset = CRDataset('/media/lscsc/nas/ziyao/DATASET/CR/DACR/changsha256/')
    dataset = ISPRSDataset(data_dir='/224010098/DATASET/DACR/Vaihingen/', mode='train', cloud_type='thick')
    dataloader = data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=8)
    for i, data in enumerate(dataloader):
        d = data
        print(i)
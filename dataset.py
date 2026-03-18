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
        self.mask_dir = getattr(cfg, 'mask_dir', os.path.join(self.data_dir, 'seg'))
        self.height_dir = getattr(cfg, 'height_dir', os.path.join(self.data_dir, 'he'))
        self.latent_dir = getattr(cfg, 'latent_dir', None)
        self.palette = {
            0: (255, 255, 255),
            1: (0, 0, 255),
            2: (0, 255, 255),
            3: (0, 255, 0),
            4: (255, 255, 0),
            5: (255, 0, 0),
            255: (0, 0, 0),
        }
        self.invert_palette = {v: k for k, v in self.palette.items()}
        self.clear_latent_db = None
        self.cloudy_latent_db = None

        if self.mode == 'train':
            img_list = np.loadtxt(os.path.join(self.data_dir, 'train.txt'), dtype=str)
        else:
            img_list = np.loadtxt(os.path.join(self.data_dir, 'test.txt'), dtype=str)
            if val_size is not None:
                img_list = img_list[:val_size]

        self.img_list = img_list

        if self.latent_dir is not None:
            clear_latent_path = os.path.join(self.latent_dir, 'clear_latent.npy')
            cloudy_latent_path = os.path.join(self.latent_dir, f'{self.cloud_type}_latent.npy')

            if os.path.exists(clear_latent_path):
                self.clear_latent_db = np.load(clear_latent_path, allow_pickle=True).item()
            if os.path.exists(cloudy_latent_path):
                self.cloudy_latent_db = np.load(cloudy_latent_path, allow_pickle=True).item()

    def __len__(self):
        return len(self.img_list)

    def convert_from_color(self, arr_3d):
        arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)
        for color, label in self.invert_palette.items():
            mask = np.all(arr_3d == np.array(color).reshape(1, 1, 3), axis=2)
            arr_2d[mask] = label
        return arr_2d

    def __getitem__(self, idx):
        file_name = self.img_list[idx].replace('tif', 'png')
        clear = io.imread(os.path.join(self.data_dir, 'clear', file_name)).astype(np.float32)[:, :, :3]
        cloudy = io.imread(os.path.join(self.data_dir, self.cloud_type, file_name)).astype(np.float32)[:, :, :3]

        clear = (clear / 255) * 2 - 1
        cloudy = (cloudy / 255) * 2 - 1

        clear = clear.transpose(2, 0, 1)
        cloudy = cloudy.transpose(2, 0, 1)

        mask = io.imread(os.path.join(self.mask_dir, file_name))
        if mask.ndim == 3:
            mask = self.convert_from_color(mask)
        mask = mask.astype(np.int64)

        height = io.imread(os.path.join(self.height_dir, file_name)).astype(np.float32)
        if height.ndim == 3:
            height = height[:, :, 0]

        data = {'clear': clear,
                'cloudy': cloudy,
                'cond': cloudy,
                'mask': mask,
                'height': height,
                'filename': self.img_list[idx]
                }

        if self.clear_latent_db is not None and file_name in self.clear_latent_db:
            data['clear_latent'] = self.clear_latent_db[file_name]
        if self.cloudy_latent_db is not None and file_name in self.cloudy_latent_db:
            data['cloudy_latent'] = self.cloudy_latent_db[file_name]

        return data


if __name__ == '__main__':
    from types import SimpleNamespace
    from torch.utils import data

    cfg = SimpleNamespace(
        data_dir='/224010098/DATASET/DACR/Vaihingen/',
        cloud_type='thick',
        mask_dir='/224010098/DATASET/DACR/Vaihingen/seg',
        height_dir='/224010098/DATASET/DACR/Vaihingen/he',
        latent_dir='./fm_feature/Vaihingen/dinov3_vitl16_sat493m',
    )

    dataset = ISPRSDataset(cfg, mode='train', val_size=4)
    dataloader = data.DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)

    batch = next(iter(dataloader))
    print('batch keys:', list(batch.keys()))
    for key, value in batch.items():
        if hasattr(value, 'shape'):
            print(f'{key}: shape={tuple(value.shape)}, dtype={value.dtype}')
        else:
            print(f'{key}: type={type(value)}')

import os
from torch.utils.data import Dataset
import numpy as np

import skimage.io as io
from pathlib import Path

PALETTE = {0: (255, 255, 255),  # Impervious surfaces (white)
           1: (0, 0, 255),  # Buildings (blue)
           2: (0, 255, 255),  # Low vegetation (cyan)
           3: (0, 255, 0),  # Trees (green)
           4: (255, 255, 0),  # Cars (yellow)
           5: (255, 0, 0),  # Clutter (red)
           255: (0, 0, 0)}  # Undefined (black)

INVERT_PALETTE = {v: k for k, v in PALETTE.items()}

def convert_to_color(arr_2d, palette=PALETTE):
    """ Numeric labels to RGB-color encoding """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)
    if arr_2d.dtype != np.uint8:
        arr_2d = arr_2d.detach().cpu().numpy().astype(np.uint8)
    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d

def convert_from_color(arr_3d, palette=INVERT_PALETTE):
    """ RGB-color encoding to grayscale labels """
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d

class ISPRSDataset(Dataset):
    def __init__(self, cfg, mode='train', val_size=None):
        super(ISPRSDataset, self).__init__()
        assert cfg.cloud_type in ['thin', 'thick']

        self.data_dir = cfg.data_dir
        self.mask_dir = cfg.mask_dir
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

        mask_color = io.imread(os.path.join(self.mask_dir, self.img_list[idx]))
        mask = convert_from_color(mask_color)

        data = {'clear': clear,
                'cloudy': cloudy,
                'cond': cloudy,
                'mask': mask,
                'filename': self.img_list[idx]
                }
        return data

if __name__ == '__main__':
    from torch.utils import data
    from types import SimpleNamespace

    cfg = {'cloud_type': 'thick',
           'data_dir': "/224010098/DATASET/DACR/Vaihingen/",
           'mask_dir': "/224010098/DATASET/DACR/Vaihingen/seg/",
           'txt_dir': "/224010098/DATASET/DACR/Vaihingen/",}
    cfg = SimpleNamespace(**cfg)

    dataset = ISPRSDataset(cfg, mode='train')
    dataloader = data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=8)
    for i, data in enumerate(dataloader):
        d = data
        print(i)
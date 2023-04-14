import os
import time
import random
import torch
import numpy as np
from torch.utils.data import Dataset
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import proj3d

from utils import collate_fn, data_prepare

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def worker_init_fn(worker_id):
    random.seed(manual_seed + worker_id)

class TOScanNet(Dataset):
    """
    Args:
        split (str): 'train' or 'test' split.
        data_root (str): path to the root directory containing data in npz format.
        split_root (str): path to the directory containing train and test splits.
        voxel_size (float): size of the voxel in meters.
        voxel_max (float): maximum number of voxels allowed.
        transform (callable, optional): A function/transform that takes in the coordinates, color, and label
            and returns a transformed version. Default: None.
        shuffle_index (int, optional): Index for shuffling the points. Default: False.
        loop (int, optional): Number of times to loop over the data. Default: 1.
        repeat_align (bool, optional): Repeat alignment? Default: False.

    Attributes:
        data_list (list): List containing names of all files in the split.
        data_idx (numpy array): Array containing the indices of the data.
        remapped (numpy array): Maps label indices to the actual segmentation category values.
    """
    def __init__(self, split: str='train', data_root: str='trainval', split_root: str="list", \
                 voxel_size: float=0.004, voxel_max: float=80000, transform=None, \
                 shuffle_index: int=False, loop: int=1, repeat_align: bool=False):
        super().__init__()
        self.split, self.voxel_size, self.transform, self.voxel_max, self.shuffle_index, \
        self.loop, self.repeat_align, self.data_root = split, voxel_size, transform, voxel_max, shuffle_index, \
            loop, repeat_align, data_root

        # get the data_list - all scenes listed inside the training set
        data_list = os.path.join(split_root, f"{split}.txt")
        with open(data_list) as d_list:
            data_list = d_list.read().splitlines()
        
        self.data_list = data_list
        self.data_idx = np.arange(len(self.data_list))

        self.remapper = np.ones(300) * (255)
        for i, x in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39, 41, \
                            42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, \
                            60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, \
                            78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 92, 93]):
            self.remapper[x] = i

        print(f"Totally {len(self.data_idx)} samples in {split} set.")

    def __getitem__(self, idx):
        data_idx = self.data_idx[idx % len(self.data_idx)]
        data_path = os.path.join(self.data_root, self.data_list[data_idx] + '.npz')
        f = np.load(data_path)
        coord, feat, label = f['xyz'], f['color'], np.expand_dims(f['semantic_label'], axis=-1)
        label = self.remapper[label.astype(int)]
        coord, feat, label = data_prepare(coord, feat, label, self.split, self.voxel_size, \
                self.voxel_max, self.transform, self.shuffle_index, self.repeat_align)
        
        return coord, feat, label

    def __len__(self):
        return len(self.data_idx) * self.loop


if __name__ == '__main__':
    DATA_ROOT = 'data/TO-scannet/train'
    SPLIT_ROOT = 'data/meta_data/TO-scannet'

    point_data = TOScanNet(split='train', data_root=DATA_ROOT, split_root=SPLIT_ROOT)
    print('point data size:', len(point_data))

    manual_seed = 123
    set_seed(manual_seed)

    train_loader = torch.utils.data.DataLoader(point_data, batch_size=16, shuffle=False, \
                                               num_workers=0, pin_memory=True, collate_fn=collate_fn)
    for idx in range(1):
        end = time.time()
        voxel_num = []
        for i, (coord, feat, label, offset) in enumerate(train_loader):
            print(f'time: {i+1}/{len(train_loader)}--{time.time() - end}')
            print('tag', coord.shape, feat.shape, label.shape, offset.shape, torch.unique(label))
            if torch.isnan(coord).any() or torch.isnan(feat).any() or torch.isnan(label).any():
                print("!")
                exit(0)
            if torch.isinf(coord).any() or torch.isinf(feat).any() or torch.isinf(label).any():
                print("!")
                exit(0)
            voxel_num.append(label.shape[0])

            # Plot for visualization
            # x = coord[:, 0]
            # y = coord[:, 1]
            # z = coord[:, 2]
            # fig = plt.figure(figsize=(8, 8))
            # ax = fig.add_subplot(111, projection='3d')

            # ax.scatter(x, y, z, marker='.', s=0.1, c=label)
            # plt.show()
            
            end = time.time()
    print(np.sort(np.array(voxel_num)))

from torch.utils.data import Dataset
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from utils.file_io import read_hdf5
from utils.utils import read_text_lines
from utils import utils

class DepthDataset(Dataset):
    def __init__(self,
                 args,
                 mode='train',
                 transform=None,
                 augmentation=None):
        super(DepthDataset, self).__init__()

        self.args = args
        self.mode = mode
        self.transform = transform
        self.augmentation = augmentation
        self.min_depth = args.min_depth
        self.max_depth = args.max_depth
        self.experimental = args.experimental
        
        hypersim_dict = {
            'train': 'dataloader/filenames/hypersim_train.txt',
            'val': 'dataloader/filenames/hypersim_val.txt',
            'test': 'dataloader/filenames/hypersim_test.txt'
        }

        self.samples = []
        data_filename = hypersim_dict[mode]
        lines = read_text_lines(data_filename)
        
        for line in lines:
            splits = line.split()
            if len(splits) == 2:
                img_name,  depth_name = splits[0], splits[1]
            else:
                raise Exception("Incorrect Data Format")

            sample = dict()
            sample["img_name"] = img_name
            sample["depth_name"] = depth_name

            self.samples.append(sample)

    def __getitem__(self, index):
        names = self.samples[index]
        sample = dict()
        img = read_hdf5(names["img_name"])
        img[img<0] = 0
        img[img>1] = 1
        sample["img"] = img[:,:,1] # G channel
        
        depth = read_hdf5(names["depth_name"]).squeeze()*1000 # convert to mm
        # depth = linear_scale(depth, self.min_depth, self.max_depth)
        scale = max(min(self.max_depth/np.percentile(depth, 99.9), 1), 0.01)
        depth = scale * depth
        sample["depth"] = depth
        
        sample["img_name"] = names["img_name"]
        sample["depth_name"] = names["depth_name"]

        if self.transform is not None:
            sample = self.transform(sample)      
                
        if self.augmentation is not None:
            # seed forces consistency between augmentations
            seed = torch.randint(0,9999999999,(1,1)).item()
            retry = 0

            if sample["depth"] is not None:
                torch.manual_seed(seed)
                depth_candidate = self.augmentation(sample["depth"])
                while torch.mean((depth_candidate == 0).float()) > 0.2 and retry <= 2:
                    retry += 1
                    seed = torch.randint(0,9999999999,(1,1)).item()
                    torch.manual_seed(seed)
                    depth_candidate = self.augmentation(sample["depth"])
            torch.manual_seed(seed)
            sample["img"] = self.augmentation(sample["img"])
            if sample["depth"] is not None:
                torch.manual_seed(seed)
                sample["depth"] = self.augmentation(sample["depth"])
            if "mask" in sample.keys():
                torch.manual_seed(seed)
                sample["mask"] = self.augmentation(sample["mask"])
            if "edges" in sample.keys():
                torch.manual_seed(seed)
                sample["edges"] = self.augmentation(sample["edges"])

        return sample

    def __len__(self):
        return len(self.samples)

# linearly scale the depth
def linear_scale(depth, min_depth, max_depth):
    data_max = np.percentile(depth, 99.9)
    data_min = np.percentile(depth, 0.1)
    a = (max_depth - min_depth)/(data_max - data_min)
    b = min_depth - a * data_min
    depth = a*depth + b
    depth[depth <= min_depth] = 0
    depth[depth >= max_depth] = 0
    
    return depth

# scale to largest contiguous slice of depth
def histogram_scale(depth, min_depth, max_depth, cutoff=0.1, recursed=0):
    hist, hist_depths = np.histogram(depth[::8,::8], 100)
#     plt.plot(hist)
    hist = np.convolve(hist, np.ones(10), mode="same") # smooth
    hist -= (hist.max()*cutoff)
    hist[hist<0] = 0
    
    bindices_zero = (hist == 0)
    indices_zero = np.arange(len(hist))[bindices_zero]
    indices_zero = np.concatenate((indices_zero, np.array([0, len(hist)-1])))
    hist_max_idx = np.argmax(hist)

    lower_bound_idx = np.max(indices_zero[np.where(indices_zero <= hist_max_idx)])
    upper_bound_idx = np.min(indices_zero[np.where(indices_zero >= hist_max_idx)])
    lower_bound_depth = hist_depths[lower_bound_idx]
    upper_bound_depth = hist_depths[upper_bound_idx]
    
    if lower_bound_idx == upper_bound_idx and recursed < 2: # avoid division by zero
        depth = histogram_scale(depth, min_depth, max_depth, cutoff/2, recursed=recursed+1)
    elif lower_bound_idx == upper_bound_idx and recursed >= 2: # don't recurse too much
        depth = depth - lower_bound_depth
        depth = min_depth + (depth * ((max_depth-min_depth)/(np.percentile(depth, 90))))
    else:
        depth = depth - lower_bound_depth
        depth = min_depth + (depth * ((max_depth-min_depth)/(upper_bound_depth-lower_bound_depth)))
        
    depth[depth <= min_depth] = 0
    depth[depth >= max_depth] = 0
    
    return depth


import torch
import numpy as np

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample

class ToTensor(object):
    """Convert numpy array to torch tensor"""
    def __init__(self, resize=None):
        self.resize = resize

    def __call__(self, sample):
        sample["img"] =  torch.from_numpy(sample["img"]).unsqueeze(0) # [1, H, W]
        if self.resize is not None:
            sample["img"] =  torch.nn.functional.interpolate(sample["img"].unsqueeze(0), self.resize, mode="bicubic", align_corners=False).squeeze(0)
        sample["depth"] =  torch.from_numpy(sample["depth"]).unsqueeze(0) # [1, H, W]
        if self.resize is not None:
            sample["depth"] =  torch.nn.functional.interpolate(sample["depth"].unsqueeze(0), self.resize, mode="bicubic", align_corners=False).squeeze(0)
        if "mask" in sample.keys():
            sample["mask"] =  torch.from_numpy(sample["mask"]).unsqueeze(0) # [1, H, W]
            if self.resize is not None:
                sample["mask"] =  torch.nn.functional.interpolate(sample["mask"].unsqueeze(0), self.resize, mode="bicubic", align_corners=False).squeeze(0)
            sample["mask"] = sample["mask"].bool()

        return sample

class RGBtoGray(object):
    """Convert lightfield array to grayscale"""
    
    def __call__(self, sample, R_weight=0.2125, G_weight=0.7154, B_weight=0.0721):
        sample["lightfield"] =  R_weight*sample["lightfield"][...,0] + \
                                G_weight*sample["lightfield"][...,1] + \
                                B_weight*sample["lightfield"][...,2]
        return sample
    
class RGBtoNIR(object):
    """Convert lightfield array to near infra-red"""
    
    def __call__(self, sample):
        interm = np.maximum(sample["lightfield"], 1-sample["lightfield"])[...,::-1]
        nir = (interm[..., 0]*0.229 + interm[..., 1]*0.587 + interm[..., 2]*0.114)**(1/0.25)
        sample["lightfield"] = nir
        return sample
    
class ToRandomPatch(object):
    """Convert full image tensors to random patch"""
    
    def __init__(self, patch_size, random_rotation=True, random_flip=True, center=False):
        self.patch_size = patch_size
        self.random_rotation = random_rotation # if true apply 0-3 90 degree rotations
        self.random_flip = random_flip # if true apply random vertical flip
        self.center = center
        
    def __call__(self, sample):
        img = sample["img"]
        depth = sample["depth"]
        
        C, H, W = img.shape
        
        if self.center:
            patch_x_coord = W//2 - (self.patch_size//2)
            patch_y_coord = H//2 - (self.patch_size//2)
        else:
            patch_x_coord = np.random.randint(0, W - self.patch_size)
            patch_y_coord = np.random.randint(0, H - self.patch_size)
    
        x1, x2 = patch_x_coord, patch_x_coord + self.patch_size
        y1, y2 = patch_y_coord, patch_y_coord + self.patch_size
        
        img_patch = img[:,y1:y2,x1:x2]
        depth_patch = depth[:,y1:y2,x1:x2]
        
        if self.random_rotation:
                rot = np.random.randint(0,4)
                img_patch = torch.rot90(img_patch, rot, dims=(1,2))
                depth_patch = torch.rot90(depth_patch, rot, dims=(1,2))
        if self.random_flip:
            if np.random.randint(0,2):
                img_patch = img_patch.flip(1)
                depth_patch = depth_patch.flip(1)
        
        sample["img_patch"] = img_patch
        sample["depth_patch"] = depth_patch
        return sample


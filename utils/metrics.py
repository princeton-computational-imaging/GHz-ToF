import numpy as np
from scipy import signal

import torch
import torch.nn as nn
import torch.nn.functional as F
# from skimage.metrics import structural_similarity as ssim

# stolen from https://github.com/andrewekhalel/sewar/blob/master/sewar/full_ref.py

def mae (GT, P):
    return np.mean(np.abs(GT-P))

def mse (GT,P):
	"""calculates mean squared error (mse).
	:param GT: first (original) input image.
	:param P: second (deformed) input image.
	:returns:  float -- mse value.
	"""
	return np.mean((GT.astype(np.float64)-P.astype(np.float64))**2)

def rmse (GT,P):
	"""calculates root mean squared error (rmse).
	:param GT: first (original) input image.
	:param P: second (deformed) input image.
	:returns:  float -- rmse value.
	"""
	return np.sqrt(mse(GT,P))

def psnr (GT,P,MAX=None):
	"""calculates peak signal-to-noise ratio (psnr).
	:param GT: first (original) input image.
	:param P: second (deformed) input image.
	:param MAX: maximum value of datarange (if None, MAX is calculated using image dtype).
	:returns:  float -- psnr value in dB.
	"""
	if MAX is None:
		MAX = np.iinfo(GT.dtype).max

	mse_value = mse(GT,P)
	if mse_value == 0.:
		return np.inf
	return 10 * np.log10(MAX**2 /mse_value)

def ssim (GT,P):
    return ssim(GT, P)

def thres_metric(GT, P, mask, thres_min, thres_max, use_np=False):
    if use_np:
        e = np.abs(GT - P)
    else:
        e = torch.abs(GT - P)
    err_mask = (e > thres_min) * (e < thres_max) * mask

    if use_np:
        mean = np.mean(err_mask.astype('float'))/np.mean(mask)
    else:
        mean = torch.mean(err_mask.float())/torch.mean(mask)

    return mean.item()

class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]


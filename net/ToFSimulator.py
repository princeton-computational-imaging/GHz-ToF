import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.utils
from utils.tof import *

def spatial_gradient(x):
    diag_down = torch.zeros_like(x)
    diag_down[:,:,1:,1:] = x[:, :, 1:, 1:] - x[:, :, :-1, :-1]
    dv = torch.zeros_like(x)
    dv[:, :, 1:, :] = x[:, :, 1:, :] - x[:, :, :-1, :]
    dh = torch.zeros_like(x)
    dh[:, :, :, 1:] = x[:, :, :, 1:] - x[:, :, :, :-1]
    diag_up = torch.zeros_like(x)
    diag_up[:, :, :-1, 1:] = x[:, :, :-1, 1:] - x[:, :, 1:, :-1]
    return [dh, dv, diag_down, diag_up]

# simulate time of flight measurements + noise model
class Simulator():
    def __init__(self, args):
        super().__init__()
        self.args = args

    def simulate(self, img, depth, val=False):
        phi_list = []
        amplitude_list = []

        # main loop
        for f in self.args.f_list:
            amplitudes = sim_quad(depth, f, self.args.T, self.args.g, img).squeeze(2) # [B, 4, H, W]
            _, amplitude_est, _ = decode_quad(amplitudes, self.args.T, self.args.mT)
            # white gaussian noise
            if val:
                noise = torch.normal(std=self.args.AWGN_sigma, mean=0, size=amplitudes.shape, 
                                     device=amplitudes.device, dtype=torch.float32, generator=torch.cuda.manual_seed(42))  
            else:
                noise_scale = torch.zeros(amplitudes.shape[0], device=amplitudes.device).uniform_(0.75,1.25)[:,None,None,None] # [B*num_patch, 1,1,1]
                noise = torch.normal(std=self.args.AWGN_sigma, mean=0, size=amplitudes.shape, 
                                     device=amplitudes.device, dtype=torch.float32)
                noise = noise * torch.sqrt(noise_scale) # random scale for training

            amplitudes += noise
            amplitudes = torch.clamp(amplitudes, 0)
            amplitudes = torch.poisson(amplitudes)
            phi_est, _, _ = decode_quad(amplitudes, self.args.T, self.args.mT)

            phi_list.append(torch.fmod(phi_est, 2*np.pi))
            amplitude_list.append(amplitude_est)

        phi_list = torch.cat(phi_list, dim=1)  # [B, # of freq, H, W]
        
        return phi_list, torch.cat(amplitude_list, dim=1)
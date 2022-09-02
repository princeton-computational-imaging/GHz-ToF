import torch.nn as nn
from utils.tof import *
from net.ToFSimulator import Simulator
from net.arch import Arch

class PhaseToPhaseNet(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.args = args
        # variable module import to select architecture
        self.arch = Arch(args)
        self.simulator = Simulator(args)
    
    # start from image + depth, simulate ToF measurements
    def forward(self, img, depth, val=False):
        phi_list, _ = self.simulator.simulate(img, depth, val)
        recon = self.arch(phi_list) 
        return recon # [B, 1, H, W]
    
    # process phase + amplitudes directly
    def process_phi(self, phi_list):
        recon = self.arch(phi_list) 
        return recon
        
import time
import torch
import torch.nn.functional as F
import numpy as np
from scipy.constants import speed_of_light
from itertools import product, combinations

from skimage.restoration import unwrap_phase

# mutates
def single_freq_unwrap(phi):
    device = phi.device
    B,C,H,W = phi.shape
    for i in range(B):
        for j in range(C):
            phi[i,j,...] = torch.from_numpy(unwrap_phase(phi[i,j,...].detach().cpu().numpy())).to(device)
    return phi



def depth_to_wraps(depth, f_list):
    wraps_split = []
    for f in f_list:
        wavelength = 1000*speed_of_light/f
        wraps = depth // (0.5*wavelength)
        wraps_split.append(wraps)
    return wraps_split

def get_max_wraps(args):
    max_f = max(args.f_list)
    wavelength = 1000*speed_of_light/max_f # mm
    max_wraps = (args.max_depth-args.min_depth)//(0.5*wavelength) + 1 # just to be safe
    
    return int(max_wraps)

def get_min_wraps(args):
    max_f = min(args.f_list)
    wavelength = 1000*speed_of_light/max_f # mm
    min_wraps = (args.max_depth-args.min_depth)//(0.5*wavelength) + 2 # just to be safe
    
    return int(min_wraps)

def sim_quad(depth, f, T, g, e): # convert
    """Simulate quad amplitude for 3D time-of-flight cameras
    Args:
        depth (tensor [B,1,H,W]): [depth map]
        f (scalar): [frequency in Hz]
        T (scalar): [integration time. metric not defined]
        g (scalar): [gain of the sensor. metric not defined]
        e (tensor [B,1,H,W]): [number of electrons created by a photon incident to the sensor. metric not defined]

    Returns:
        amplitudes (tensor [B,4,H,W]): [Signal amplitudes at 4 phase offsets.]
    """
    
    tru_phi = depth2phase(depth, f)
    
    A0 = g*e*(0.5+torch.cos(tru_phi)/np.pi)*T
    A1 = g*e*(0.5-torch.sin(tru_phi)/np.pi)*T
    A2 = g*e*(0.5-torch.cos(tru_phi)/np.pi)*T
    A3 = g*e*(0.5+torch.sin(tru_phi)/np.pi)*T
    
    return torch.stack([A0, A1, A2, A3], dim=1)


def decode_quad(amplitudes, T, mT): # convert
    """Simulate solid-state time-of-flight range camera.
    Args:
        amplitudes (tensor [B,4,H,W]): [Signal amplitudes at 4 phase offsets. See sim_quad().]
        T (scalar): [Integration time. Metric not defined.]
        mT (scalar): [Modulation period]

    Returns:
        phi_est, amplitude_est, offset_est (tuple(tensor [B,1,H,W])): [Estimated phi, amplitude, and offset]
    """
    assert amplitudes.shape[1] % 4 == 0
    
    A0, A1, A2, A3 = amplitudes[:,0::4,...], amplitudes[:,1::4,...], amplitudes[:,2::4,...], amplitudes[:,3::4,...]
    sigma = np.pi * T / mT
    
    phi_est = torch.atan2((A3-A1),(A0-A2))
    phi_est[phi_est<0] = phi_est[phi_est<0] + 2*np.pi
    
    amplitude_est = (sigma/T*np.sin(sigma)) * (( (A3-A1)**2 + (A0-A2)**2 )**0.5)/2 
    offset_est = (A0+A1+A2+A3)/(4*T)

    return phi_est, amplitude_est, offset_est

def depths2phases(depth, f_list):
    phis = []
    for i, f in enumerate(f_list):
        tru_phi = (4*np.pi*depth*f)/(1000*speed_of_light)
        phis.append(tru_phi)
    return torch.cat(phis, dim=1)

def depth2phase(depth, freq): # convert
    """Convert depth map to phase map.
    Args:
        depth (tensor [B,1,H,W]): Depth map (mm)
        freq (scalar): Frequency (hz)

    Returns:
        phase (tensor [B,1,H,W]): Phase map (radian)
    """

    tru_phi = (4*np.pi*depth*freq)/(1000*speed_of_light)
    return tru_phi

def phase2depth(phase, freq): # convert
    """Convert phase map to depth map.
    Args:
        phase (tensor [B,1,H,W]): Phase map (radian)
        freq ([type]): Frequency (Hz)

    Returns:
        depth (tensor [B,1,H,W]): Depth map (mm)
    """

    depth = (1000*speed_of_light*phase)/(4*np.pi*freq)
    return depth

def CRT_get_prod(f_list, min_depth, max_depth):
    # Compute the range of potential n (wraps)
    max_f = max(f_list)
    
    min_phase = depth2phase(min_depth, max_f)
    max_phase = depth2phase(max_depth, max_f)
    min_wraps = int(min_phase//(2*np.pi))
    max_wraps = int(max_phase//(2*np.pi))
        
    prod = []
    from itertools import permutations
    prod = list(permutations(np.arange(max_wraps), 2))
    
#     for i in range(min_wraps, max_wraps):
#         Q = i*(1/max_f) # approx num wavelengths of the smallest carrier wave
#         C1 = (int(Q/(1/f_list[0])), int(Q/(1/f_list[1])))
#         C2 = (int(Q/(1/f_list[0])), max(int(Q/(1/f_list[1]))-1, 0))
#         C3 = (int(Q/(1/f_list[0])), min(int(Q/(1/f_list[1]))+1, max_wraps-1))
#         C4 = (int(Q/(1/f_list[0])), max(int(Q/(1/f_list[1]))-2, 0))
#         C5 = (int(Q/(1/f_list[0])), min(int(Q/(1/f_list[1]))+2, max_wraps-1))
#         prod.append(C1)
#         prod.append(C2)
#         prod.append(C3)
#         prod.append(C4)
#         prod.append(C5)
        
    prod = torch.tensor(prod)
    prod = torch.unique(prod, dim=0)
    
    return prod

def CRT_get_prod_count(f_list, min_depth, max_depth):
    prod = CRT_get_prod(f_list, min_depth, max_depth)
    max_wraps = torch.max(prod).item()
    
    return max_wraps, len(prod)

def CRT_unwrap(phi, f_list, min_depth=0, max_depth=2500):

    """Efficient Multi-Frequency Phase Unwrapping using Kernel Density Estimation
    Args:
        phi (list(tensor [B,C,H,W])): C different wrapped phases measured at the given frequencies f_list
        f_list (list [C]): C different frequencies in Hz.
        min_depth (scalar) : min depth in mm
        max_depth (scalar) : max depth in mm

    Returns:
        depth (tensor [B,1,H,W]): Unwrapped depth map (mm)
    """
    assert len(f_list) == 2 # only implemented for 2 freqs
    assert f_list[1] >= f_list[0] # sorted
    
    err, prod = CRT_get_err(phi, f_list, min_depth, max_depth)
    depth = CRT_process_err(err, phi, f_list, min_depth, max_depth)

    return depth

def CRT_get_err(phi, f_list, min_depth=0, max_depth=2500):
    assert len(f_list) == 2 # only implemented for 2 freqs
    assert f_list[1] >= f_list[0] # sorted
    
    B,num_f,H,W = phi.shape
    device = phi.device
    
    # Compute the range of potential n (wraps)
    max_f = max(f_list)
    
    min_phase = depth2phase(min_depth, max_f)
    max_phase = depth2phase(max_depth, max_f)
    min_wraps = int(min_phase//(2*np.pi))
    max_wraps = int(max_phase//(2*np.pi))
    
#     n_list = [list(range(0,max_wraps//2)), list(range(0,max_wraps))]

#     from itertools import product, combinations
#     prod = list(product(*n_list))
        
    prod = []
    from itertools import permutations
    prod = list(permutations(np.arange(max_wraps), 2))
#     for i in range(min_wraps, max_wraps):
#         Q = i*(1/max_f) # approx num wavelengths of the smallest carrier wave
#         C1 = (int(Q/(1/f_list[0])), int(Q/(1/f_list[1])))
#         C2 = (int(Q/(1/f_list[0])), max(int(Q/(1/f_list[1]))-1, 0))
#         C3 = (int(Q/(1/f_list[0])), min(int(Q/(1/f_list[1]))+1, max_wraps-1))
#         C4 = (int(Q/(1/f_list[0])), max(int(Q/(1/f_list[1]))-2, 0))
#         C5 = (int(Q/(1/f_list[0])), min(int(Q/(1/f_list[1]))+2, max_wraps-1))
#         prod.append(C1)
#         prod.append(C2)
#         prod.append(C3)
#         prod.append(C4)
#         prod.append(C5)
        
    prod = torch.tensor(prod, device=device)
    prod = torch.unique(prod, dim=0)

    num_prod = len(prod) # number of potential combinations

    k = torch.tensor(np.lcm.reduce(np.array(f_list).astype(np.int))/f_list, device=device)

    t = []
    for i in range(num_f):
        t.append( (phi[:,i,:,:]/(2*np.pi))*k[i] )
    t = torch.stack(t) # [num_f, B, H, W]

    err = torch.zeros((B,num_prod,H,W), device=device)
    for i in range(num_prod):
        err_i = (k[0]*prod[i][0] - k[1]*prod[i][1] - (t[1] - t[0]))**2
        err[:,i,:,:] += err_i
    
    return err, prod

def CRT_process_err(err, phi, f_list, min_depth, max_depth):
    device = err.device
    prod = CRT_get_prod(f_list, min_depth, max_depth).to(device)
    num_f = prod.shape[1]
    num_prod,B,H,W = err.shape
    ind_min = torch.argmin(err, dim=1)

    # index into wrap counts at optimal indices
    prod_array = prod[:,:,None,None].repeat(1,1,H,W)
    
    n_list_best = torch.stack([prod_array[:,0,...].gather(0, ind_min),
                               prod_array[:,1,...].gather(0, ind_min)])
    n_list_best = n_list_best.squeeze(1)

    # simple phase unwrapping with the ranking
    unwrapped_phase = []
    for i in range(num_f):
        unwrapped_phase.append(phi[:,i,:,:]+2*np.pi*n_list_best[i,...])
    unwrapped_phase = torch.stack(unwrapped_phase)    

    depth = []
    for i in range(num_f):
        depth_i = phase2depth(unwrapped_phase[i], f_list[i])
        depth.append(depth_i)

    depth = torch.stack(depth, dim=1)
    
    return depth.to(device)


def positional_encoding(tensor, num_encoding_functions=6, include_input=True, scale_phi = False, log_sampling=True):
    r"""Apply positional encoding to the input.
    Args:
        tensor (torch.Tensor): Input tensor to be positionally encoded.
        encoding_size (optional, int): Number of encoding functions used to compute
            a positional encoding (default: 6).
        include_input (optional, bool): Whether or not to include the input in the
            positional encoding (default: True).
    Returns:
    (torch.Tensor): Positional encoding of the input tensor.
    """
    # Trivially, the input tensor is added to the positional encoding.
    if scale_phi:
        encoding = [tensor/(2*np.pi)] if include_input else []
    else:
        encoding = [tensor] if include_input else []
    frequency_bands = None
    if log_sampling:
        frequency_bands = 2.0 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    else:
        frequency_bands = torch.linspace(
            2.0 ** 0.0,
            2.0 ** (num_encoding_functions - 1),
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    for freq in frequency_bands:
        for func in [torch.sin, torch.cos]:
            encoding.append(func(tensor * freq))

    # Special case, for no positional encoding
    if len(encoding) == 1:
        return encoding[0]
    else:
        return torch.cat(encoding, dim=1)
    
def soft_argmax(x, temp=1e2):
    """
    Arguments: x [torch.tensor], input class weights (B, C, H, W)
    Returns: [torch.tensor], output differentiable argmax result (B, 1, H, W)
    """
    
    B,C,H,W = x.shape
    # optionally multiply x by some large weight before softmax 
    # to push the value closer to a real argmax output, though this might
    # not bode well for your gradients
    x = F.softmax(x * temp, dim=1) 
    indices = torch.arange(0, C, device=x.device)[None,:,None,None] # [B, C, H, W]
    x = torch.sum(x*indices, dim=1, keepdim=True)
    
    return x
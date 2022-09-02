import numpy as np
import re
import h5py

def read_pfm(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header.decode("ascii") == 'PF':
        color = True
    elif header.decode("ascii") == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode("ascii").rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale

def read_calib_txt(file):
    # hard-coded for middlebury v3 dataset
    with open(file, "r") as f:
        focal = float(f.readline().split(" ")[0][6:])
        _ = f.readline()
        doffs = float(f.readline().split("=")[1])
        baseline = float(f.readline().split("=")[1])
    
    return baseline, focal, doffs

def RGB_to_NIR(img):
    interm = np.maximum(img, 1-img)[...,::-1]
    nir = (interm[..., 0]*0.229 + interm[..., 1]*0.587 + interm[..., 2]*0.114)**(1/0.25)
    return nir

def read_hdf5(file):
    with h5py.File(file, "r") as f:
        key = list(f.keys())[0]
        data = np.array(f[key], dtype=np.float32)
        data = np.nan_to_num(data, nan=0.0)
    return data
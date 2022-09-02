import torch
import torchvision
from torch.utils.data import DataLoader
from importlib.machinery import SourceFileLoader

import argparse
import numpy as np
import os

from dataloader.dataloader import DepthDataset
from dataloader import transforms
from utils import utils
from utils import tof
from net import model
from net.PhaseToPhaseNet import PhaseToPhaseNet
from importlib.machinery import SourceFileLoader
parser = argparse.ArgumentParser()

# Training args
parser.add_argument("--mode", default="train", type=str, help="Network mode [train, val, test]")
parser.add_argument("--checkpoint_dir", type=str, required=True, help="Directory to save model checkpoints and logs")
parser.add_argument('--pretrained_net', default=None, type=str, help='Pretrained network')

parser.add_argument("--batch_size", default=16, type=int, help="Batch size for training")
parser.add_argument("--val_batch_size", default=16, type=int, help="Val batch size for training")
parser.add_argument("--num_workers", default=12, type=int, help="Number of workers for data loading")
parser.add_argument("--seed", default=60221409, type=int, help="Seed for PyTorch/NumPy.")
parser.add_argument("--weight_decay", default=0, type=float, help="Weight decay for optimizer")
parser.add_argument("--max_epoch", default=1000, type=int, help="Maximum number of epochs for training")
parser.add_argument("--iter_per_epoch", default=300, type=int, help="Iterations/batches per epoch.")
parser.add_argument("--val_iter_per_epoch", default=3, type=int, help="Iterations/batches per epoch (validation).")

parser.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint")
parser.add_argument("--no_validate", action="store_true", help="No validation")

parser.add_argument('--print_freq', default=600, type=int, help='Print frequency to screen (# of iterations)')
parser.add_argument('--summary_freq', default=600, type=int, help='Summary frequency to tensorboard (# of iterations)')
parser.add_argument('--val_freq', default=5, type=int, help='Validation frequency (# of epochs)')
parser.add_argument('--save_ckpt_freq', default=10, type=int, help='Save checkpoint frequency (# of epochs)')

# Image-specific
parser.add_argument('--patch_size', default=512, type=int, help='Image patch size for patch-based training.')
parser.add_argument("--downsize", action="store_true", help='downsize the whole image')
parser.add_argument("--max_wrap", default=57, type=int, help="Scale to this max depth (mm).")
parser.add_argument("--min_wrap", default=36, type=int, help="Assume this min depth (mm).")

# Network-specific
parser.add_argument("--f_list", default="7.150e9,14.320e9", type=str, help="List of modulation frequencies for phase unwrapping.")
parser.add_argument('--f_unwrap', default="min", type=str, choices=['min','max'], help="Use min or max frequency for unwrapping.")
parser.add_argument("--g", default=20, type=float, help="Gain of the sensor. Metric not defined.")
parser.add_argument("--T", default=1000, type=float, help="Integration time. Metric not defined.")
parser.add_argument("--mT", default=2000, type=float, help="Modulation period. Default 2x integration time.")
parser.add_argument("--AWGN_sigma", default=1200, type=float, help="Additive white gaussian noise's standard deviation.")
parser.add_argument("--lr", default=1e-3, type=float, help="Network learning rate")
parser.add_argument("--gamma", default=0.999, type=float, help="Rate of decay for ExponentialLR")
parser.add_argument("--num_encoding_functions", default=6, type=int, help="Number of encoding functions for fourier features.")
parser.add_argument("--L1_weight", default=1, type=float, help="Weight for L1 loss on phi")
parser.add_argument("--CE_weight", default=1, type=float, help="Weight for CE loss on wrap")
parser.add_argument("--nf", default=32, type = int, help="Starting size of filter")

# experimental data
parser.add_argument("--experimental", action="store_true", help="Train for use on experimental data.")

args = parser.parse_args()
# adjust based on batch_size
args.print_freq //= args.batch_size
args.summary_freq //= args.batch_size
args.iter_per_epoch //= args.batch_size

# torch.autograd.set_detect_anomaly(True)
args.f_list = [float(f) for f in args.f_list.split(",")]

args.num_prod = 1
if args.f_unwrap == "min": # minimize number of classes
    args.min_depth = tof.phase2depth(args.min_wrap*2*np.pi, min(args.f_list))
    args.max_depth = tof.phase2depth(args.max_wrap*2*np.pi, min(args.f_list))
    args.max_wraps = args.max_wrap - args.min_wrap

utils.check_path(args.checkpoint_dir)
utils.save_args(args)
utils.save_net_files(args)

def main():    
    # Seed for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    # speedup if input is same size
    torch.backends.cudnn.benchmark = True
    
    print("=> Training args: {0}".format(args))
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("=> Training on {0} GPU(s)".format(torch.cuda.device_count()))
    else:
        device = torch.device("cpu")
        print("=> Training on CPU")

    # Train loader
    if args.downsize:
            train_transform = transforms.Compose([transforms.ToTensor(resize=(384,512))])
    else:
        train_transform = transforms.Compose([transforms.ToTensor(resize=None)])

    train_augmentation = transforms.Compose([torchvision.transforms.RandomCrop((args.patch_size,args.patch_size), padding=0)])
    train_data = DepthDataset(args, mode="train", transform=train_transform, augmentation=train_augmentation)
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=False)

    # Validation loader
    if args.downsize:
        val_transform = transforms.Compose([transforms.ToTensor(resize=(384,512))])
    else:
        val_transform = transforms.Compose([transforms.ToTensor(resize=None)])
        
    val_augmentation = transforms.Compose([torchvision.transforms.CenterCrop((args.patch_size,args.patch_size))])
    val_data = DepthDataset(args, mode="val", transform=val_transform, augmentation=val_augmentation)
    val_loader = DataLoader(dataset=val_data, batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    
    print("=> {} training samples found in the training set".format(len(train_data)))
    
    # Network
    if args.pretrained_net is not None:
        print("=> Loading pretrained network: %s" % args.pretrained_net)
        # Enable training from a partially pretrained model
        net = torch.load(os.path.join(args.pretrained_net, "full_net_latest.pt"), map_location=device)
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        net.args = args
        virtual_arch = SourceFileLoader("Arch", os.path.join(args.pretrained_net, "net_files", "arch.py")).load_module()
        net.Arch = virtual_arch.Arch(net.args)
        virtual_simulator = SourceFileLoader("Simulator", os.path.join(args.checkpoint_dir, "net_files", "ToFSimulator.py")).load_module()
        net.simulator = virtual_simulator.Simulator(net.args)  
        net.to(device)
    else:
        net = PhaseToPhaseNet(args, device).to(device)
        
    optimizer = torch.optim.Adam(params=net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if torch.cuda.device_count() > 1:
        print("=> Using %d GPUs" % torch.cuda.device_count())
        net = torch.nn.DataParallel(net)

    # Parameters
    num_params = utils.count_parameters(net)
    print("=> Number of trainable parameters: %d" % num_params)

    # Resume training
    if args.resume:
        # Load Network
        start_epoch, start_iter = utils.resume_latest_ckpt(args.checkpoint_dir, net, "net")
        # Load Optimizer
        utils.resume_latest_ckpt(args.checkpoint_dir, optimizer, "optimizer")
    else:
        start_epoch = 0
        start_iter = 0

    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.iter_per_epoch*args.max_epoch, pct_start=0.05, cycle_momentum=False, anneal_strategy="linear")
    train_model = model.Model(args, optimizer, lr_scheduler, net, device, start_iter, start_epoch)

    print("=> Start training...")

    for epoch in range(start_epoch, args.max_epoch):
        train_model.train(train_loader)
        if not args.no_validate:
            if epoch % args.val_freq == 0 or epoch == (args.max_epoch - 1):
                train_model.validate(val_loader)

    print("=> End training\n\n")


if __name__ == "__main__":
    main()

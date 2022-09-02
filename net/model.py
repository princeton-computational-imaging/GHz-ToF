import torch
import time
from torch.utils.tensorboard import SummaryWriter
from utils import utils
import numpy as np
import os
import utils.tof as tof


class Model(object):
    def __init__(self, args, optimizer, lr_scheduler, net, device, start_iter=0, start_epoch=0):
        self.args = args
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.net = net
        self.device = device
        self.num_iter = start_iter
        self.epoch = start_epoch
        self.train_writer = SummaryWriter(self.args.checkpoint_dir)
        self.CE_loss = torch.nn.CrossEntropyLoss()
        self.L1_loss = torch.nn.L1Loss()

    def train(self, train_loader):
        args = self.args
        self.net.train() # init Module
        last_print_time = time.time()

        for i, sample in enumerate(train_loader):
            if i >= args.iter_per_epoch:
                break
        
            self.process(sample)
            loss = self.regression_loss + self.classifier_loss
           
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
                        
            # print log
            if self.num_iter % args.print_freq == 0:
                this_cycle = time.time() - last_print_time
                last_print_time += this_cycle
                print("Epoch: [%3d/%3d] Iter: [%5d/%5d] time: %4.2fs loss: %.3f" %
                            (self.epoch + 1, args.max_epoch, i + 1, args.iter_per_epoch, this_cycle, loss.item()))
                
            # image summary for tensorboard
            if self.num_iter % args.summary_freq == 0:
                self.write_summary(is_val=False)
                
            self.num_iter += 1
            self.lr_scheduler.step()
            
        self.epoch += 1
        
        if args.no_validate:
            self.save_net()

    def validate(self, val_loader):
        args = self.args
        print("=> Start validation...")

        self.net.eval()

        num_samples = len(val_loader)
        print("=> %d samples found in the validation set" % num_samples)
        val_file = os.path.join(args.checkpoint_dir, "val_results.txt")
        
        val_loss = 0
        val_regression_loss = 0
        val_classifier_loss = 0
        valid_samples = 0

        for i, sample in enumerate(val_loader):
            if i >= args.val_iter_per_epoch:
                break
                
            self.process(sample, is_val=True)
                    
            val_loss += (self.classifier_loss + self.regression_loss)
            val_regression_loss += self.regression_loss
            val_classifier_loss += self.classifier_loss
            
            self.write_summary(is_val=True)
            valid_samples += 1
            
        print("=> Validation done!")

        val_loss = val_loss / valid_samples
        val_regression_loss = val_regression_loss / valid_samples
        val_classifier_loss = val_classifier_loss / valid_samples
        
        # Save validation results
        with open(val_file, "a") as f:
            f.write("epoch: %03d\t" % self.epoch)
            f.write("val_classifier_loss: %.3f\t" % val_classifier_loss)
            f.write("val_regression_loss: %.3f\t" % val_regression_loss)
            f.write("val_loss: %.3f\n" % val_loss)

        print("=> Mean validation loss of epoch %d: loss: %.6f" % (self.epoch, val_loss))
        self.save_net()
            
    def process(self, sample, is_val=False):
        args = self.args
        device = self.device
    
        self.img = sample["img"].to(device)  # [B, 1, H, W]
        self.depth = sample["depth"].to(device) # [B, 1, H, W]
        self.phi = torch.fmod(tof.depths2phases(self.depth, args.f_list), 2*np.pi) # wrapped phase

        if "mask" in sample.keys():
            self.mask = (self.depth > 1e-32) * (self.depth < args.max_depth) * sample["mask"].to(device) * (self.depth > args.min_depth) 
        else:
            self.mask = (self.depth > 1e-32) * (self.depth < args.max_depth) * (self.depth > args.min_depth) # ignore invalid areas
        
        ## NET CALL ##
        if is_val:
            with torch.no_grad():
                self.depth_pred, self.phi_pred, self.wrap_pred = self.net(self.img, self.depth, val=False) # [B, 1, H, W]
        else:
            self.depth_pred, self.phi_pred, self.wrap_pred = self.net(self.img, self.depth, val=False) # [B, 1, H, W]
        ## /NET CALL ##
            
        if args.f_unwrap == "max": # integer number of wraps
            self.wrap = (tof.depth2phase(torch.clamp(self.depth - args.min_depth, min=0),
                                         max(args.f_list)) // (2*np.pi)).long()
        else:
            self.wrap = (tof.depth2phase(torch.clamp(self.depth - args.min_depth, min=0),
                                         min(args.f_list)) // (2*np.pi)).long()
        self.wrap[self.wrap > args.max_wraps - 1] = args.max_wraps - 1 # clamp
        
        if args.CE_weight > 0:
            self.classifier_loss = args.CE_weight * self.CE_loss(self.wrap_pred * self.mask, self.wrap.squeeze(1) * self.mask.squeeze(1)) 
        else:
            self.classifier_loss = torch.tensor(0)

        if args.L1_weight > 0:
            self.regression_loss = args.L1_weight * self.L1_loss(self.depth * self.mask, self.depth_pred * self.mask)
        else:
            self.regression_loss = torch.tensor(0)

        if args.L1_weight == args.CE_weight == 0:
            raise Exception("No valid loss.")
    
    def write_summary(self, is_val):
        args = self.args
        flag = "val" if is_val else "train"
        
        if not is_val: # learning rate summary
            lr = self.optimizer.param_groups[0]["lr"]
            self.train_writer.add_scalar("lr", lr, self.epoch + 1)
        
        # Write tensorboard images
        img_summary = dict()
        img_summary["img"] = utils.colorize(self.img, 0, 1, "gray")
        img_summary["depth_gt"] = utils.colorize(self.mask * self.depth, 0, args.max_depth, "nipy_spectral")
        img_summary["depth_pred"] = utils.colorize(self.mask * self.depth_pred, 0, args.max_depth, "nipy_spectral")
        abs_error = self.mask * torch.abs(self.depth_pred - self.depth)
        img_summary["absolute_error"] = utils.colorize(abs_error, 0, abs_error[0].mean()*2, "hot")

        img_summary["wrap_gt"] = utils.colorize(self.mask * self.wrap, 0, args.max_wraps, "nipy_spectral")
        img_summary["wrap_pred"] = utils.colorize(self.mask * torch.argmax(self.wrap_pred, dim=1, keepdim=True), 0, args.max_wraps, "nipy_spectral")
        utils.save_images(self.train_writer, flag, img_summary, self.num_iter)

        img_summary = dict()
        
        # Write tensorboard scalars
        self.train_writer.add_scalar("{0}/loss".format(flag), self.classifier_loss.item() + self.regression_loss.item(), self.num_iter)
        self.train_writer.add_scalar("{0}/classifier_loss".format(flag), self.classifier_loss.item(), self.num_iter)
        self.train_writer.add_scalar("{0}/regression_loss".format(flag), self.regression_loss.item(), self.num_iter)
        self.train_writer.add_scalar("{0}/MAE".format(flag), torch.mean(torch.abs(self.mask * self.depth_pred - self.mask * self.depth)), self.num_iter)
        self.train_writer.add_scalar("{0}/MSE".format(flag), torch.mean((self.mask * self.depth_pred - self.mask * self.depth)**2), self.num_iter)
        
    def save_net(self):
        args = self.args
        torch.save(self.net, os.path.join(args.checkpoint_dir, "full_net_latest_epoch_{0}.pt".format(self.epoch)))
        utils.save_checkpoint(args.checkpoint_dir, self.optimizer, self.net,
                              epoch=self.epoch, num_iter=self.num_iter,
                              loss=-1, filename="net_latest.pt")
        print("model saved")

        # Save checkpoint of specific epoch
        if self.epoch % args.save_ckpt_freq == 0:
            model_dir = os.path.join(args.checkpoint_dir, "models")
            utils.check_path(model_dir)
            utils.save_checkpoint(model_dir, self.optimizer, self.net,
                                  epoch=self.epoch, num_iter=self.num_iter,
                                  loss=-1, save_optimizer=False)
        

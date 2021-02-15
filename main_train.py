import argparse
import os
import shutil
from torchvision import utils
from model.improved_video_gan import ImprovedVideoGAN
import torch
import torch.nn as nn

import torchvision

import torch.nn.parallel
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import time
import numpy as np

parser = argparse.ArgumentParser(description='VideoGAN')

parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--gpu', default='0,1', help='index of gpus to use')
parser.add_argument('--epochs', default=40, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--arch', metavar='ARCH', default='basic_fcn', type=str,
                    help='model architecture: ' + ' (default: basic_fcn)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=60, type=int,
                    metavar='N', help='mini-batch size (default: 60)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate (default: 1e-3')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-3, type=float,
                    metavar='W', help='weight decay (default: 1e-3)')
parser.add_argument('--lr_step', default='40,60', help='decreasing strategy')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on test set')
parser.add_argument('-wl', '--weighted-loss', dest='weighted_loss', action='store_true',
                    help='use weighted CE loss')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--num-frozen', default=0, type=int, metavar='N',
                    help='# frozen cnv2 layers')
parser.add_argument('--root-dir', default='/datasets/UCF-101', type=str, metavar='PATH',
                    help='directory containing videos and index file (default: /datasets/UCF-101)')
parser.add_argument('--index-file', default='mjpeg-index.txt', type=str, metavar='FILENAME',
                    help='index file referencing all videos relative to root_dir (default: mjpeg-index.txt)')
parser.add_argument('--save-dir', default='extra/', type=str, metavar='PATH',
                    help='path to directory for saved outputs (default: extra/)')
parser.add_argument('--color-sat', default=0.0, type=float,
                    help='factor for saturation jitter transform (default: 0.0)')
parser.add_argument('--color-hue', default=0.0, type=float,
                    help='factor for hue jitter transform (default: 0.0)')

def main():
    global args
    args = parser.parse_args()
    out_dir = args.save_dir

    # Check if Output Directory Exists
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    GAN = ImprovedVideoGAN(batch_size=args.batch_size, z_dim=100, critic_iterations=5)



    G = GAN.get_generator()
    D = GAN.get_discriminator()


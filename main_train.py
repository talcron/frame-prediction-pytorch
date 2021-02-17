import argparse
import os

from comet_ml import Experiment
import torch
from torch.utils.data import DataLoader

from data.dataloader import VideoDataset
from model.improved_video_gan import ImprovedVideoGAN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors


def get_parser():
    global parser
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
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        metavar='N', help='mini-batch size (default: 64)')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                        metavar='LR', help='initial learning rate (default: 1e-4')
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
    parser.add_argument('--beta1', default=0.5, type=float,
                        help='Beta parameter for ADAM (default: 0.5)')
    parser.add_argument('--zdim', default=100, type=int,
                        help='Dimensionality of hidden features (default: 100)')
    parser.add_argument('--exp-name', default='dev', type=str,
                        help='The experiment name (default: deb)')
    parser.add_argument('--exp-disable', default=False, type=bool,
                        help='Disable CometML (default: False)')
    return parser


def main(args):
    out_dir = args.save_dir
    experiment = Experiment(disabled=args.exp_disable)
    experiment.add_tag(args.exp_name)

    # Check if Output Directory Exists
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    dataset = VideoDataset(args.index_file, device=DEVICE)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )
    GAN = ImprovedVideoGAN(
        dataloader=dataloader,
        experiment=experiment,
        epoch_range=(args.start_epoch, args.epochs + args.start_epoch),
        batch_size=args.batch_size,
        learning_rate=args.lr,
        z_dim=args.zdim,
        beta1=args.beta1,
        critic_iterations=5
    )
    GAN.to(DEVICE)
    GAN.train()


if __name__ == '__main__':
    parser = get_parser()
    main(parser.parse_args())

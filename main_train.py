import argparse
import os
import sys

from comet_ml import Experiment, ExistingExperiment
import torch
from torch.utils.data import DataLoader

from data.dataloader import DatasetFactory
from runner import ImprovedVideoGAN


def get_parser():
    global parser
    parser = argparse.ArgumentParser(description='VideoGAN')

    run_args = parser.add_argument_group('Run', "Run options")
    run_args.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                          help='evaluate model on test set')
    run_args.add_argument('--root-dir', default='/', type=str, metavar='PATH',
                          help='directory containing videos and index file (default: /)')
    run_args.add_argument('--index-file', default='mjpeg-index.txt', type=str, metavar='FILENAME',
                          help='index file referencing all videos relative to root_dir (default: mjpeg-index.txt)')
    run_args.add_argument('--save-dir', default='extra/', type=str, metavar='PATH',
                          help='path to directory for saved outputs (default: extra/)')
    run_args.add_argument('--resume', default='', type=str, metavar='PATH',
                          help='path to latest checkpoint (default: none)')
    run_args.add_argument('--epochs', default=40, type=int, metavar='N',
                          help='number of total epochs to run')

    learner_args = parser.add_argument_group("Learner")
    learner_args.add_argument('--spec-norm', default=False, action='store_true',
                              help='set to True to use spectral normalization')
    learner_args.add_argument('--no-gp', default=False, action='store_true',
                              help='set to True to stop using Gradient Penalty')
    learner_args.add_argument('--drift-penalty', default=False, action='store_true',
                              help='set to True to use small drift penalty (prevents discriminator loss drifting)')
    learner_args.add_argument('--zero-centered', default=False, action='store_true',
                              help='set to True to use zero-centered GP')
    learner_args.add_argument('--one-sided', default=False, action='store_true',
                              help='set to True to use one-sided GP')

    compute_args = parser.add_argument_group("Compute")
    compute_args.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                              help='number of data loading workers (default: 0)')
    compute_args.add_argument('--num-gpu', default=1, type=int, help='number of GPUs to use')
    compute_args.add_argument('--cache-dataset', default=False, action='store_true',
                              help='Keep all data in memory (default: False if switch is absent)')

    model_args = parser.add_argument_group("Model")
    model_args.add_argument('--arch', metavar='ARCH', default='basic_fcn', type=str,
                            help='model architecture: ' + ' (default: basic_fcn)')
    model_args.add_argument('--pretrained', dest='pretrained', action='store_true',
                            help='use pre-trained model')
    model_args.add_argument('--num-frozen', default=0, type=int, metavar='N',
                            help='# frozen cnv2 layers')
    model_args.add_argument('--zdim', default=100, type=int,
                            help='Dimensionality of hidden features (default: 100)')

    optimizer_args = parser.add_argument_group("Optimizer")
    optimizer_args.add_argument('-b', '--batch-size', default=64, type=int,
                                metavar='N', help='mini-batch size (default: 64)')
    optimizer_args.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                                metavar='LR', help='initial learning rate (default: 1e-4')
    optimizer_args.add_argument('--momentum', default=0.9, type=float, metavar='M',
                                help='momentum')
    optimizer_args.add_argument('--weight-decay', '--wd', default=0.0, type=float,
                                metavar='W', help='weight decay (default: 0.0)')
    optimizer_args.add_argument('--lr_step', default='40,60', help='decreasing strategy')
    optimizer_args.add_argument('--beta1', default=0.5, type=float,
                                help='Beta parameter for ADAM (default: 0.5)')

    pre_process_args = parser.add_argument_group("Pre-process")
    pre_process_args.add_argument('--color-sat', default=0.0, type=float,
                                  help='factor for saturation jitter transform (default: 0.0)')
    pre_process_args.add_argument('--color-hue', default=0.0, type=float,
                                  help='factor for hue jitter transform (default: 0.0)')

    log_args = parser.add_argument_group("Log")
    log_args.add_argument('--print-freq', '-p', default=100, type=int,
                          metavar='N', help='print frequency (default: 10)')
    log_args.add_argument('--exp-name', default='dev', type=str, nargs='+',
                          help='The experiment name (default: deb)')
    log_args.add_argument('--exp-key', default='', type=str,
                          help='The key to an existing experiment to be continued (default: None)')
    log_args.add_argument('--exp-disable', default=False, action='store_true',
                          help='Disable CometML (default: False if switch is absent)')
    return parser


def get_experiment(args):
    if args.resume and args.exp_key:
        experiment = ExistingExperiment(
            disabled=args.exp_disable,
            previous_experiment=args.exp_key,
            log_code=True,
            log_env_gpu=True,
            log_env_cpu=True,
            log_env_details=True,
            log_env_host=True,
            log_git_metadata=True,
            log_git_patch=True,
        )
    else:
        experiment = Experiment(disabled=args.exp_disable)

    if isinstance(args.exp_name, list):
        for tag in args.exp_name:
            experiment.add_tag(tag)
    else:
        experiment.add_tag(args.exp_name)

    experiment.log_text(' '.join(sys.argv))
    return experiment


def main(args):
    experiment = get_experiment(args)
    assert os.path.exists(args.save_dir), f'save-dir {args.save_dir} does not exist'
    assert torch.cuda.device_count() >= args.num_gpu, f'You have requested more gpus than are available'

    if isinstance(args.exp_name, list):
        exp_name = '_'.join(args.exp_name)
    else:
        exp_name = args.exp_name
    out_dir = os.path.join(args.save_dir, exp_name)

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    experiment.log_text(f"{sys.executable} {' '.join(sys.argv)} --resume {os.path.join(out_dir, 'checkpoint.model')} "
                        f"--exp-key {experiment.id}")

    if args.cache_dataset and args.workers > 0:
        ResourceWarning("You are using multiple workers and keeping data in memory, this will multiply memory usage"
                        "by the number of workers.")
    dataset = DatasetFactory.get_dataset(os.path.join(args.root_dir, args.index_file), cache_dataset=args.cache_dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
    )

    GAN = ImprovedVideoGAN(
        dataloader=dataloader,
        experiment=experiment,
        device=DEVICE,
        num_gpu=args.num_gpu,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        z_dim=args.zdim,
        beta1=args.beta1,
        critic_iterations=5,
        out_dir=out_dir,
        spec_norm=args.spec_norm,
        no_gp=args.no_gp,
        drift_penalty=args.drift_penalty,
        zero_centered=args.zero_centered,
        one_sided=args.one_sided,
    )

    if args.resume != '':
        GAN.load(args.resume)

    try:
        GAN.train()
    except BaseException as e:
        GAN.save()
        raise e
    finally:
        experiment.send_notification(f"Experiment {exp_name} is complete")


if __name__ == '__main__':
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors

    parser = get_parser()
    main(parser.parse_args())

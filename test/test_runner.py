import unittest

import torch
from comet_ml import Experiment
from torch.utils.data import DataLoader

from data.dataloader import VideoDataset
from runner import ImprovedVideoGAN

BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
experiment = Experiment(disabled=True)


class TestSaving(unittest.TestCase):
    def setUp(self) -> None:
        dataset = VideoDataset('data/mjpeg/index.txt')
        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        )
        self.batch, _ = next(iter(dataloader))

        self.gan = ImprovedVideoGAN(
            dataloader=dataloader,
            experiment=experiment,
            device=DEVICE,
            num_gpu=1,
            n_epochs=1,
            batch_size=BATCH_SIZE,
            learning_rate=0,
            z_dim=100,
            beta1=0.5,
            critic_iterations=5,
            out_dir='results',
        )

    def test_save_batch_as_gif(self):
        self.gan._save_batch_as_gif(self.batch, name='real')


if __name__ == '__main__':
    unittest.main()

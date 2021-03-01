import random
import unittest

import matplotlib.pyplot as plt
import torch

from data.dataloader import VideoDataset

DATA_INDEX_FILE = 'data/mjpeg/index.txt'


class TestDataloader(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = VideoDataset(DATA_INDEX_FILE)

    def test_sample_has_correct_shape_and_normalization(self):
        num_samples = 20
        sample_indices = random.choices(range(len(self.dataset)), k=num_samples)
        for i in sample_indices:
            video, label = self.dataset[i]
            self.assertIsInstance(label, str)
            self.assertEqual((3, 32, 64, 64), video.shape)
            self.assertGreaterEqual(video.min(), -1.)
            self.assertLessEqual(video.max(), 1.)

    def test_frame_loads_correctly(self):
        video, label = self.dataset[50]
        plt.imshow(video[:, 10, ...].permute((1, 2, 0)))
        plt.show()
        worked = input('Enter y/[n]')
        self.assertIn('y', worked)


if __name__ == '__main__':
    unittest.main()

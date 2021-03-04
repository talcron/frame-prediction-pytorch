from abc import ABC, abstractmethod
import random
import unittest

import matplotlib.pyplot as plt

from data.dataloader import DatasetFactory

UCF_INDEX_FILE = 'data/mjpeg/index.txt'
TINYVIDEO_INDEX_FILE = 'data/tinyvideo/index.txt'


class TestUcfDataLoader(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = DatasetFactory.get_dataset(UCF_INDEX_FILE, dataset='UCF-101')

    def test_sample_has_correct_shape_and_normalization(self):
        num_samples = 20
        sample_indices = random.choices(range(len(self.dataset)), k=num_samples)
        for i in sample_indices:
            video, label = self.dataset[i]
            self.assertIsInstance(label, (str, type(None)))
            self.assertEqual((3, 32, 64, 64), video.shape)
            self.assertGreaterEqual(video.min(), -1.)
            self.assertLessEqual(video.max(), 1.)

    def test_frame_loads_correctly(self):
        video, label = self.dataset[20]
        video = video / 2 + 0.5  # [-1, 1] -> [0, 1]
        plt.imshow(video[:, 10, ...].permute((1, 2, 0)))
        plt.show()
        worked = input('Enter y/[n]')
        self.assertIn('y', worked)


class TestTinyvideoDataLoader(TestUcfDataLoader):
    def setUp(self) -> None:
        self.dataset = DatasetFactory.get_dataset(TINYVIDEO_INDEX_FILE, dataset='tinyvideo')


if __name__ == '__main__':
    unittest.main()

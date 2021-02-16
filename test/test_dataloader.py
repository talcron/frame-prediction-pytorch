import random
import unittest

from data.dataloader import VideoDataset


class TestDataloader(unittest.TestCase):
    def setUp(self) -> None:
        self.dataloader = VideoDataset('/home/ian/Videos/UCF-101/mjpeg-index.txt')

    def test_sample(self):
        num_samples = 20
        sample_indices = random.choices(range(len(self.dataloader)), k=num_samples)
        for i in sample_indices:
            video, label = self.dataloader[i]
            self.assertIsInstance(label, str)
            self.assertEqual((32, 64, 64, 3), video.shape)
            self.assertGreaterEqual(video.min(), -1.)
            self.assertLessEqual(video.max(), 1.)


if __name__ == '__main__':
    unittest.main()

from functools import lru_cache

import numpy as np
import torch
from skvideo.io import vread
from torch.utils.data import Dataset

UCF101 = "UCF-101"
UCF_SPORTS = "ucf_sports"
DATASETS = {UCF101,
            UCF_SPORTS}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors


class VideoDataset(Dataset):

    def __init__(self, index_file, num_frames=32, shape=(64, 64), dataset=UCF101, device=DEVICE, cache_dataset=False):
        """

        Args:
            index_file:
            num_frames:
            shape:
            dataset:
            device:
            cache_dataset: if true, store the uint8 videos in memory.
        """
        assert dataset in DATASETS, f"{dataset} not in {DATASETS}"
        self.device = device
        self.dataset = dataset
        self.num_frames = num_frames
        self.shape = shape
        self.keep_in_memory = cache_dataset
        self.data = self._read_file(index_file)

        self._read_video = self._get_read_video_func(cache_dataset)

    @staticmethod
    def _get_read_video_func(keep_in_memory):
        def _read_video(fn):
            video = vread(fn, inputdict={'-s': '64x64'}, num_frames=32)
            return video

        if keep_in_memory:
            return lru_cache()(_read_video)
        else:
            return _read_video

    def stats(self):
        # todo: verify a consistent timebase between videos. The UCF videos are all 1:10
        filenames = self.data
        stats = ('width', 'height', 'n_frames')
        stats = {s: np.zeros((len(filenames),)) for s in stats}
        for i, fn in enumerate(filenames):
            video = self._read_video(fn)
            n_frames, height, width, channels = video.shape
            stats['n_frames'][i] = n_frames
            stats['width'][i] = width
            stats['height'][i] = height

        print("Video Metadata:")
        for k, v in stats.items():
            print("{} | max: {} | min: {}".format(k, v.max(), v.min()))

        return stats

    @staticmethod
    def _read_file(index_file):
        with open(index_file, 'r') as f:
            data = f.read().split('\n')[:-1]
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fn = self.data[idx]
        video = torch.from_numpy(self._read_video(fn)).to(self.device)
        video = video.to(self.device)
        video = video.permute((3, 0, 1, 2))  # (frame, channels, height, width)

        video = (video / 127.) - 1.  # normalize to the range [-1, 1]
        label = self._label_from_path(fn)
        return video, label

    def _label_from_path(self, fn):
        if self.dataset == UCF101:
            label = fn.split('/')[-2]
        elif self.dataset == UCF_SPORTS:
            label = fn.split('/')[-3]
        else:
            raise ValueError(f"{self.dataset} not in {DATASETS}")
        return label

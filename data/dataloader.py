from functools import lru_cache

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

FFMPEG_INPUT_DICT = {
    '-s': '64x64',       # width x height
    # '-r': '10',            # fps
    # '-pix_fmt': 'vdpau_mpeg4'  # todo: look this up
}
FFMPEG_OUTPUT_DICT = {
    '-s': '64x64',       # width x height
    # '-r': '10',            # fps
    # '-pix_fmt': 'vdpau_mpeg4'  # todo: look this up
}

UCF101 = "UCF-101"
UCF_SPORTS = "ucf_sports"
DATASETS = {UCF101,
            UCF_SPORTS}


# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors


class VideoDataset(Dataset):

    def __init__(self, index_file, num_frames=32, shape=(64, 64), dataset=UCF101, cache_dataset=False):
        """

        Args:
            index_file: Path to the text file where each line has a file path to a video
            num_frames: The number of frames to load per video
            shape: The frame shape. Much match video encoding
            dataset: The dataset type
            cache_dataset: if true, store the uint8 videos in memory.
        """
        assert dataset in DATASETS, f"{dataset} not in {DATASETS}"
        self.dataset = dataset
        self.num_frames = num_frames
        self.shape = shape
        self.keep_in_memory = cache_dataset
        self.data = self._read_file(index_file)

        self._read_video = self._get_read_video_func(cache_dataset)
        self._transform = transforms.Compose([
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def _get_read_video_func(self, keep_in_memory):
        def read_video(fn):
            reader = cv2.VideoCapture(fn)
            video = np.zeros((self.num_frames, *self.shape, 3), dtype=np.uint8)
            for i in range(self.num_frames):
                _, frame = reader.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video[i] = frame
            video = torch.from_numpy(video)
            return video

        if keep_in_memory:
            return lru_cache()(read_video)
        else:
            return read_video

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
        video = self._read_video(fn)
        video = video.permute((3, 0, 2, 1))  # todo: verify this (channels, frames, height, width)

        video = (video / (255. / 2)) - 1.  # normalize to the range [-1, 1]
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

from abc import ABC, abstractmethod
from functools import lru_cache

import cv2
import numpy as np
import torch
import torchvision.transforms
from torch.utils.data import Dataset
import torch.nn.functional as F
from PIL import Image

UCF101 = "UCF-101"
UCF_SPORTS = "ucf_sports"
TINYVIDEO = "tinyvideo"
DATASETS = {UCF101,
            UCF_SPORTS,
            TINYVIDEO}


class SafeDataset(Dataset):
    """A wrapper around a torch.utils.data.Dataset that allows dropping
    samples dynamically.

    https://github.com/msamogh/nonechucks/blob/master/nonechucks/dataset.py
    """

    def __init__(self, dataset, eager_eval=False):
        """Creates a `SafeDataset` wrapper around `dataset`."""
        self.dataset = dataset
        self.eager_eval = eager_eval
        # These will contain indices over the original dataset. The indices of
        # the safe samples will go into _safe_indices and similarly for unsafe
        # samples.
        self._safe_indices = []
        self._unsafe_indices = []

        # If eager_eval is True, we can simply go ahead and build the index
        # by attempting to access every sample in self.dataset.
        if self.eager_eval is True:
            self._build_index()

    def _safe_get_item(self, idx):
        """Returns None instead of throwing an error when dealing with an
        unsafe sample, and also builds an index of safe and unsafe samples as
        and when they get accessed.
        """
        try:
            # differentiates IndexError occuring here from one occuring during
            # sample loading
            invalid_idx = False
            if idx >= len(self.dataset):
                invalid_idx = True
                raise IndexError
            sample = self.dataset[idx]
            if idx not in self._safe_indices:
                self._safe_indices.append(idx)
            return sample
        except Exception as e:
            if isinstance(e, IndexError):
                if invalid_idx:
                    raise
            if idx not in self._unsafe_indices:
                self._unsafe_indices.append(idx)
            return None

    def _build_index(self):
        for idx in range(len(self.dataset)):
            # The returned sample is deliberately discarded because
            # self._safe_get_item(idx) is called only to classify every index
            # into either safe_samples_indices or _unsafe_samples_indices.
            _ = self._safe_get_item(idx)

    def _reset_index(self):
        """Resets the safe and unsafe samples indices."""
        self._safe_indices = self._unsafe_indices = []

    @property
    def is_index_built(self):
        """Returns True if all indices of the original dataset have been
        classified into safe_samples_indices or _unsafe_samples_indices.
        """
        return len(self.dataset) == len(self._safe_indices) + len(self._unsafe_indices)

    @property
    def num_samples_examined(self):
        return len(self._safe_indices) + len(self._unsafe_indices)

    def __len__(self):
        """Returns the length of the original dataset.
        NOTE: This is different from the number of actually valid samples.
        """
        return len(self.dataset)

    def __iter__(self):
        return (
            self._safe_get_item(i)
            for i in range(len(self))
            if self._safe_get_item(i) is not None
        )

    def __getitem__(self, idx):
        """Behaves like the standard __getitem__ for Dataset when the index
        has been built.
        """
        while idx < len(self.dataset):
            sample = self._safe_get_item(idx)
            if sample is not None:
                return sample
            idx += 1
        raise IndexError

    def __getattr__(self, key):
        """Delegates to original dataset object if an attribute is not
        found in this class.
        """
        return getattr(self.dataset, key)


class DatasetFactory:
    @classmethod
    def get_dataset(cls, index_file, **kwargs):
        """

        Args:
            index_file: Path to the text file where each line has a file path to a video

        Returns:
            Dataset based on the dataset type
        """
        class_map = {
            UCF101: UcfDataset,
            UCF_SPORTS: UcfDataset,
            TINYVIDEO: TinyvideoDataset,
        }

        dataset = kwargs.get('dataset', None)
        dataset = dataset if dataset is not None else cls._infer_dataset_from_path(index_file)
        ds = class_map[dataset](index_file, **kwargs)
        return SafeDataset(ds)

    @staticmethod
    def _infer_dataset_from_path(path):
        for ds in DATASETS:
            if ds in path:
                return ds
        raise RuntimeError("Could not infer dataset type from path.")


class VideoDataset(Dataset, ABC):
    def __init__(self, index_file, **kwargs):
        """

        Args:
            index_file: Path to the text file where each line has a file path to a video
            num_frames: The number of frames to load per video
            shape: The frame shape. Much match video encoding
            cache_dataset: if true, store the uint8 videos in memory.
        """
        self.unnormalized_max = 255.
        self.num_frames = kwargs.get('num_frames', 32)
        self.shape = kwargs.get('shape', (64, 64))
        self.keep_in_memory = kwargs.get('cache_dataset', False)
        self.data = self._read_file(index_file)

        self._read_video = self._get_read_video_func()

    def _get_read_video_func(self):
        if self.keep_in_memory:
            return lru_cache()(self._read_video)
        else:
            return self._read_video

    @staticmethod
    def _read_file(index_file):
        with open(index_file, 'r') as f:
            data = f.read().split('\n')[:-1]
        return data

    def __len__(self):
        return len(self.data)

    def normalize(self, video):
        return (video / (self.unnormalized_max / 2.)) - 1.

    @staticmethod
    def un_normalize(video):
        return ((video + 1.) * (255. / 2.)).byte()

    @abstractmethod
    def _read_video(self, fn):
        ...

    @abstractmethod
    def _label_from_path(self, fn):
        ...

    def __getitem__(self, item):
        fn = self.data[item]
        video = self._read_video(fn)

        video = self.normalize(video)
        label = self._label_from_path(fn)
        return video, label


class UcfDataset(VideoDataset):
    def __init__(self, *args, dataset=UCF101, **kwargs):
        self.dataset = dataset
        super(UcfDataset, self).__init__(*args, **kwargs)

    def stats(self):
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

    def _read_video(self, fn):
        reader = cv2.VideoCapture(fn)
        video = np.zeros((self.num_frames, *self.shape, 3), dtype=np.uint8)
        for i in range(self.num_frames):
            success = reader.read(video[i])
            if not success or np.all(video[i] == 0):
                print(f"No frame read from {fn}, frame {i}")
                if i > 0:  # tile the rest of the frames
                    video[i:] = video[i - 1]
                    break
                else:
                    raise Exception(f"No frames could be read from {fn}")
            cv2.cvtColor(video[i], cv2.COLOR_BGR2RGB, video[i])
        video = torch.from_numpy(video)
        video = video.permute((3, 0, 1, 2))
        return video

    def _label_from_path(self, fn):
        if self.dataset == UCF101:
            label = fn.split('/')[-2]
        elif self.dataset == UCF_SPORTS:
            label = fn.split('/')[-3]
        else:
            label = None
        return label


class TinyvideoDataset(VideoDataset):
    def __init__(self, *args, **kwargs):
        super(TinyvideoDataset, self).__init__(*args, **kwargs)
        self.unnormalized_max = 1.
        self.to_tensor = torchvision.transforms.ToTensor()
        self.to_pil = torchvision.transforms.ToPILImage()

    def _read_video(self, fn: str) -> torch.Tensor:
        video = Image.open(fn)
        if video.size[0] != self.shape[0]:
            video = self._downscale_video(video, fn)
        else:
            video = self.to_tensor(video)
            video = video.reshape([3, self.num_frames, *self.shape])
        return video

    def _downscale_video(self, video: Image, fn: str) -> torch.Tensor:
        size = video.size[0]
        video = self.to_tensor(video)
        n_frames = video.shape[1] // size
        video = video.reshape([3, -1, size, size])
        if n_frames < self.num_frames:
            new_video = torch.zeros([3, self.num_frames, size, size])
            new_video[:, :n_frames] = video
            new_video[:, n_frames:] = torch.unsqueeze(video[:, -1], dim=1)
            video = new_video
        video = video[:, :self.num_frames]
        video = F.interpolate(video, size=self.shape)
        try:
            self.to_pil(video.reshape([3, self.num_frames * self.shape[0], self.shape[1]])).save(fn)
        except PermissionError:
            pass

        return video

    def _label_from_path(self, fn):
        """
        These videos are unlabeled, so we return an empty string.
        """
        return ''

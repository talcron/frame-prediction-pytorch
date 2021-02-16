import numpy as np
import torchvision.transforms as transforms
from skvideo.io import vread
from torch.utils.data import Dataset

DATASETS = {UCF101 := "UCF-101",
            UCF_SPORTS := "ucf_sports"}


class VideoDataset(Dataset):

    def __init__(self, index_file, num_frames=32, shape=(64, 64), dataset=UCF101):
        assert dataset in DATASETS, f"{dataset} not in {DATASETS}"
        self.dataset = dataset
        self.num_frames = num_frames
        self.shape = shape
        self.data = self._read_file(index_file)

        self.transform = self._define_transformation()

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
    def _read_video(fn):
        # noinspection PyTypeChecker
        video = vread(fn, inputdict={'-s': '64x64'}, num_frames=32)
        return video

    @staticmethod
    def _define_transformation():
        to_tensor = transforms.ToTensor()

        resnet_normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        standard_normalize = transforms.Normalize((0, 0, 0, 0), (1, 1, 1, 1))
        return transforms.Compose([to_tensor, standard_normalize])

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

        video = self.transform(video)
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

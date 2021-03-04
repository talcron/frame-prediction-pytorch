"""
Takes tarfile as input and extracts and processes videos.
Converts jpeg frames into 64x64 mjpeg videos with max 32 frames
"""

import argparse
import os
import tarfile

import torchvision

parser = argparse.ArgumentParser()
parser.add_argument('file')
args = parser.parse_args()

root = os.path.dirname(args.file)
transform = torchvision.transforms.Resize([64, 64])


def downscale(path):
    try:
        video = torchvision.io.read_image(path)
        video = video.reshape([-1, 3, 128, 128])
        video = transform(video)
        video = video[:32].reshape([3, 32*64, 64])
        torchvision.io.write_jpeg(video, filename=path)
    except RuntimeError:
        print(f"!!!skipped {path}")
        os.remove(path)


def do_stuff(members):
    last_jpg = None
    for tarinfo in members:
        if last_jpg is not None:
            downscale(os.path.join(root, last_jpg))
            last_jpg = None
        if tarinfo.name.endswith('.mp4'):
            print(tarinfo.name)
        elif tarinfo.name.endswith('.jpg'):
            last_jpg = tarinfo.name
        yield tarinfo


tar = tarfile.open(args.file)
tar.extractall(path=root, members=do_stuff(tar))

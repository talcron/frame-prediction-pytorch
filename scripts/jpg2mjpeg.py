import argparse
import glob
import os

parser = argparse.ArgumentParser()
parser.add_argument('--root-dir', required=True)
args = parser.parse_args()

files = glob.iglob(f'{args.root_dir}/**/*.mp4', recursive=True)

for path in files:
    avi_path = f'{path[:-4]}.avi'
    mjpeg_path = f'{path[:-4]}.mjpeg'

    # Skip mjpegs that have already been converted
    if os.path.exists(mjpeg_path):
        continue

    # Convert jpegs to AVI and delete the images
    cmd = f"ffmpeg -y -f image2 -i {path}/%04d.jpg {avi_path}"
    print(cmd)
    os.system(cmd)

    # convert AVI to MJPEG and delete the AVI
    cmd = f"ffmpeg -i {avi_path} -c:v mjpeg -vf scale=64:64 {mjpeg_path} -y"
    print(cmd)
    os.system(cmd)
    os.remove(avi_path)

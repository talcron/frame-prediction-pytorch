# WGAN Video Generation

The `fvd` module incorporates code from [Google Research](https://github.com/google-research/google-research), 
licensed under the Apache 2.0 license, the text of which can be found in the LICENSE file. 

The model is implemented in PyTorch and adapted from the [iVGAN](https://github.com/bernhard2202/improved-video-gan) Tensorflow implementation.

![Generated Sample 1](../media/wgan_gp_fake_3_188k.gif?raw=true)
![Generated Sample 3](../media/zero_centered_fake_1_68k.gif?raw=true)

## Setup

Requires:

```
pytorch
torchvision
tensorflow-gpu
tensorflow-hub
tensorflow-gan
comet_ml
matplotlib
opencv
```

### Environment

#### Conda

We recommend using a `conda` environment. Once you have conda installed (look at 
[miniconda](https://docs.conda.io/en/latest/miniconda.html)). Install with:

```
conda create env -f environment.yml
conda activate torch
```

#### Docker

Get the Docker image used to run this:

```
docker pull ianpegg9/torch:tf
```

The included `Dockerfile` contains the spec for this image.

### CometML

To use CometML to log experimental results, create a `.comet.config` file with contents like this:
```ini
[comet]
api_key=MYKEY
project_name=frame-prediction-pytorch
workspace=username
```

Otherwise, pass the argument `--exp-disable` to the script.

### Data

Our dataloaders support UCF-101, UCF-sports, and tinyvideo.

#### Tinyvideo

[Tinyvideo](http://www.cs.columbia.edu/~vondrick/tinyvideo/#data)

We recommend downloading 'Beach only' or 'Golf only'. Regardless of what you download, it should be saved in a tarball
and saved in a directory named 'tinyvideo'.

Pre-process everything to 64x64 to save space with this huge dataset. 
Use a machine with many processors, or it will take several days to process the entire tarball.

```
python scripts/process_tinyvideo.py <path-to-data-tarball>
```

Create the index file:

```
find /path/to/tinyvideo -name *.jpg > /path/to/tinyvideo/index.txt
```

#### UCF-101 and UCF sports

[UCF](https://www.crcv.ucf.edu/data/UCF101.php)

Download and unzip the data file.

Pre-process everything for easy loading. We don't delete the original videos with this pre-processor.

```
bash scripts/process_ucf.sh /path/to/ucf-data
```

Create the index file:

```
find /path/to/ucf-data -name *.mjpeg > /path/to/ucf-data/index.txt
```

## Usage

The sections below describe the main use cases. For further options, use `python main_train.py --help`

### Train

`--root-dir /path/to/data/dir`

`--index-file root-dir/index.txt`

`--save-dir /where/you/save/your/results`

`--exp-name <tag or tags for CometML>`

### Continue training

`--resume /path/to/checkpoint.model`

Optional:
 
`--exp-key <comet-ml experiment key>` to continue a CometML experiment.

### Evaluate

Evaluate instead of training.

`--evaluate`

Real data to compare to:

`--index-file root-dir/index.txt`

Checkpoint to load:

`--resume /path/to/checkpoint.model`

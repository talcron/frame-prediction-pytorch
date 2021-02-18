from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.utils.data
import numpy as np

DISCRIMINATOR = 'discriminator'
GENERATOR = 'generator'
BETA2 = 0.999
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors


def no_grad(model: nn.Module):
    """
    A decorator that
    Args:
        model:

    Returns:

    """
    def decorator_no_grad(func):
        def wrapper_no_grad(*args, **kwargs):
            model.requires_grad_(False)
            retval = func(*args, **kwargs)
            model.requires_grad_(True)
            return retval

        return wrapper_no_grad

    return decorator_no_grad


class ImprovedVideoGAN(object):
    def __init__(self,
                 dataloader,
                 experiment,
                 device=DEVICE,
                 epoch_range=(0, 50),
                 batch_size=64,
                 num_frames=32,
                 crop_size=64,
                 learning_rate=0.0002,
                 z_dim=100,
                 beta1=0.5,
                 critic_iterations=5):
        self.device = device
        self.epoch_range = epoch_range
        self.dataloader: torch.utils.data.DataLoader = dataloader
        self.critic_iterations = critic_iterations
        self.crop_size = crop_size
        self.beta1 = beta1
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.z_dim = z_dim
        self.num_frames = num_frames

        self.generator = self.get_generator()
        self.discriminator = self.get_discriminator()

        self.discriminator_optimizer = self._make_optimizer(self.discriminator)
        self.generator_optimizer = self._make_optimizer(self.generator)

        self._experiment = experiment
        self._log_parameters()

    def _log_parameters(self):
        self._experiment.log_parameters({
            'epoch_range': self.epoch_range,
            'batch_size': self.batch_size,
            'num_frames': self.num_frames,
            'crop_size': self.crop_size,
            'learning_rate': self.learning_rate,
            'z_dim': self.z_dim,
            'beta1': self.beta1,
            'critic_iterations': self.critic_iterations,
        })

    def _make_optimizer(self, model):
        """
        Create an Adam optimizer for the parameters of the provided model

        Args:
            model: Uses the parameters of this model in the optimizer

        Returns:
            an optimizer
        """
        return torch.optim.Adam(
            model.parameters(),
            lr=self.learning_rate,
            betas=(self.beta1, BETA2),
        )

    def to(self, device):
        """
        Moves generator and discriminator models to the specified device and updates model's device

        Args:
            device: device to move tensors to

        Returns:
            None
        """
        self.device = device
        self.generator.to(device)
        self.discriminator.to(device)

    def get_generator(self):
        """
        Create and initialize a generator model

        Returns:
            generator model
        """
        g_net = Generator(self.z_dim)
        g_net.apply(init_weights)
        return g_net

    def get_discriminator(self):
        """
        Create and initialize a discriminator model

        Returns:
            discriminator model
        """
        d_net = Discriminator()
        d_net.apply(init_weights)
        return d_net

    def train(self):
        """
        Main training loop. Run this to train the models.
        """
        for epoch in range(*self.epoch_range):
            self._experiment.set_epoch(epoch)
            for step, batch in enumerate(self.dataloader):
                self._experiment.set_step(step)
                if (step + 1) % self.critic_iterations:
                    self.optimize(batch[0], DISCRIMINATOR)
                else:
                    self.optimize(batch[0], GENERATOR)
            self._experiment.log_epoch_end(epoch)

    def optimize(self, batch, model_type):
        """
        Run optimization

        Args:
            batch: batch of real videos
            model_type: "generator" or "discriminator"

        Returns:
            None
        """
        assert model_type in {GENERATOR, DISCRIMINATOR}
        z_vec = torch.rand((self.batch_size, self.z_dim), device=self.device)
        fake_videos = self.generator(z_vec)

        if model_type == GENERATOR:
            with self.discriminator.no_grad:
                self._optimize_generator(fake_videos)
        elif model_type == DISCRIMINATOR:
            with self.generator.no_grad:
                self._optimize_discriminator(batch, fake_videos.detach())

    def _optimize_discriminator(self, batch, fake_videos):
        d_fake = self.discriminator(fake_videos)
        d_real = self.discriminator(batch)
        g_cost = -torch.mean(d_fake)
        d_cost = -g_cost - torch.mean(d_real)

        self._experiment.log_metric('g_cost', g_cost)
        self._experiment.log_metric('d_cost', d_cost)

        alpha = torch.rand(size=(self.batch_size, 1, 1, 1, 1), device=self.device)
        interpolates = batch + (alpha * (fake_videos - batch))
        interpolates.requires_grad_(True)
        interpolates.retain_grad()
        d_hat = self.discriminator(interpolates)
        with self.discriminator.no_grad:
            d_hat.backward(torch.ones_like(d_hat), retain_graph=True)

        gradients = interpolates.grad
        slopes = torch.sqrt(torch.sum(torch.square(gradients), dim=1))
        # noinspection PyTypeChecker
        gradient_penalty = torch.mean((slopes - 1.) ** 2)
        d_cost_final = d_cost + 10 * gradient_penalty

        d_cost_final.backward()
        self.discriminator_optimizer.step()
        print(f'discriminator cost: {d_cost_final}')
        self._experiment.log_metric('d_cost_final', d_cost_final)
        self.discriminator_optimizer.zero_grad()

    def _optimize_generator(self, fake_videos):
        d_fake = self.discriminator(fake_videos)
        g_cost = -torch.mean(d_fake)
        g_cost.backward()

        self.generator_optimizer.step()
        print(f'generator cost: {g_cost}')
        self._experiment.log_metric('g_cost', g_cost)
        self.generator_optimizer.zero_grad()


class BaseGAN(nn.Module):
    @property
    @contextmanager
    def no_grad(self):
        self.requires_grad_(False)
        yield
        self.requires_grad_(True)


class Generator(BaseGAN):
    """
    Generator for improved_video_gan model

    z is the dimension of the Gaussian noise input into the generator
    the linear layer output is reshaped into 512 channels of 4x4 2-frame videos
    Each conv layer then halves the # of channels while doubling the # of frames and spatial size

    Args:
        z_dim: the dimension of the encoded image
    """

    def __init__(self, z_dim):
        super(Generator, self).__init__()
        # Linear Block
        self.linear = nn.Linear(z_dim, 512 * 4 * 4 * 2, bias=True)
        self.bn0 = nn.BatchNorm3d(512, affine=True)

        # Conv Block 1
        self.deconv1 = nn.ConvTranspose3d(in_channels=512, out_channels=256,
                                          kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm3d(256, affine=True)

        # Conv Block 2
        self.deconv2 = nn.ConvTranspose3d(in_channels=256, out_channels=128,
                                          kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm3d(128, affine=True)

        # Conv Block 3
        self.deconv3 = nn.ConvTranspose3d(in_channels=128, out_channels=64,
                                          kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm3d(64, affine=True)

        # Conv Block 4
        self.deconv4 = nn.ConvTranspose3d(in_channels=64, out_channels=3,
                                          kernel_size=4, stride=2, padding=1)

        # Activation Functions
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.linear(x)
        x = torch.reshape(x, [-1, 512, 2, 4, 4])
        x = self.relu(self.bn0(x))

        x = self.relu(self.bn1(self.deconv1(x)))

        x = self.relu(self.bn2(self.deconv2(x)))

        x = self.relu(self.bn3(self.deconv3(x)))

        out = self.tanh(self.deconv4(x))

        return out


class Discriminator(BaseGAN):
    """Discriminator for improved_video_gan model"""

    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=3, out_channels=64,
                               kernel_size=4, stride=2, padding=1)
        self.ln1 = nn.LayerNorm([64, 16, 32, 32])

        self.conv2 = nn.Conv3d(in_channels=64, out_channels=128,
                               kernel_size=4, stride=2, padding=1)
        self.ln2 = nn.LayerNorm([128, 8, 16, 16])

        self.conv3 = nn.Conv3d(in_channels=128, out_channels=256,
                               kernel_size=4, stride=2, padding=1)
        self.ln3 = nn.LayerNorm([256, 4, 8, 8])

        self.conv4 = nn.Conv3d(in_channels=256, out_channels=512,
                               kernel_size=4, stride=2, padding=1)
        self.ln4 = nn.LayerNorm([512, 2, 4, 4])

        self.conv5 = nn.Conv3d(in_channels=512, out_channels=1,
                               kernel_size=4, stride=2, padding=1)

        self.linear = nn.Linear(in_features=4, out_features=1, bias=True)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.leaky_relu(self.ln1(self.conv1(x)))
        x = self.leaky_relu(self.ln2(self.conv2(x)))
        x = self.leaky_relu(self.ln3(self.conv3(x)))
        x = self.leaky_relu(self.ln4(self.conv4(x)))
        x = self.leaky_relu(self.conv5(x))
        x = torch.reshape(x, [x.shape[0], -1])
        out = self.linear(x)

        return out


# Not exactly sure what type of initialization this is (looks like Xavier ??) but it's what the
# paper uses
def _init_conv3d(m):
    fan_in = m.in_channels * 4 * 4 * 4
    fan_out = m.out_channels * 2 * 2 * 2
    std_val = 2. * (np.sqrt(3) / (fan_in + fan_out))
    nn.init.uniform_(m.weight, a=-std_val, b=std_val)


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0.0, std=0.01)
        nn.init.zeros_(m.bias)
    elif type(m) == nn.ConvTranspose3d:
        _init_conv3d(m)
    elif type(m) == nn.Conv3d:
        _init_conv3d(m)

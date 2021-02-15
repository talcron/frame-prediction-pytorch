import torch
import torch.nn as nn
import numpy as np


class ImprovedVideoGAN(object):
    def __init__(self,
                 batch_size=64,
                 num_frames=32,
                 crop_size=64,
                 learning_rate=0.0002,
                 z_dim=100,
                 beta1=0.5,
                 alpha1=0.1,
                 critic_iterations=5):
        self.critic_iterations = critic_iterations
        self.crop_size = crop_size
        self.beta1 = beta1
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.z_dim = z_dim
        self.num_frames = num_frames
        self.alpha1 = alpha1

    def get_generator(self):
        g_net = Generator(self.z_dim)
        g_net.apply(init_weights)
        return g_net

    def get_discriminator(self):
        d_net = Discriminator()
        d_net.apply(init_weights)
        return d_net


class Generator(nn.Module):
    """Generator for improved_video_gan model"""

    def __init__(self, z):
        super(Generator, self).__init__()

        # z is the dimension of the Gaussian noise input into the generator
        # the linear layer output is reshaped into 512 channels of 4x4 2-frame videos
        # Each conv layer then halves the # of channels while doubling the # of frames and spatial size
        # Linear Block
        self.linear = nn.Linear(z, 512 * 4 * 4 * 2, bias=True)
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


class Discriminator(nn.Module):
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

        self.leakyrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)


    def forward(self, x):
        x = self.leakyrelu(self.ln1(self.conv1(x)))
        x = self.leakyrelu(self.ln2(self.conv2(x)))
        x = self.leakyrelu(self.ln3(self.conv3(x)))
        x = self.leakyrelu(self.ln4(self.conv4(x)))
        x = self.leakyrelu(self.conv5(x))
        x = torch.reshape(x, [1, -1])
        out = self.linear(x)

        return out


# class Custom_Loss(nn.Module):
#     """Must return a scalar variable"""
#     def __init__(self):
#         super(Custom_Loss, self).__init__()
#
#     def forward(self, outputs, labels):


# Actually I think all we need is 2-3 nn.MSEloss instances w different weights
criterion = nn.MSELoss()

# Not exactly sure what type of initialization this is (looks like Xavier ??) but it's what the
# paper uses
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0.0, std=0.01)
        nn.init.zeros_(m.bias)
    elif type(m) == nn.ConvTranspose3d:
        fan_in = m.in_channels * 4 * 4 * 4
        fan_out = m.out_channels * 2 * 2 * 2
        std_val = 2. * (np.sqrt(3) / (fan_in + fan_out))
        nn.init.uniform_(m.weight, a=-std_val, b=std_val)
    elif type(m) == nn.Conv3d:
        fan_in = m.in_channels * 4 * 4 * 4
        fan_out = m.out_channels * 2 * 2 * 2
        std_val = 2. * (np.sqrt(3) / (fan_in + fan_out))
        nn.init.uniform_(m.weight, a=-std_val, b=std_val)


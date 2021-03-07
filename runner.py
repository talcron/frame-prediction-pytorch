import os

import torch
import torch.cuda
import torchvision.utils

from model.improved_video_gan import Generator, init_weights, Discriminator
from utils.functional import frechet_inception_distance

CHECKPOINT_FILENAME = 'checkpoint.model'

SAVE_INTERVAL = 1000     # steps
SAMPLE_INTERVAL = 100  # steps
FID_INTERVAL = 100    # steps
GRADIENT_MULTIPLIER = 10.


DISCRIMINATOR = 'discriminator'
GENERATOR = 'generator'
BETA2 = 0.999
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors


class ImprovedVideoGAN(object):
    def __init__(
            self,
            dataloader,
            experiment,
            device=DEVICE,
            n_epochs=50,
            num_gpu=1,
            batch_size=64,
            num_frames=32,
            crop_size=64,
            learning_rate=0.0002,
            weight_decay=0.0,
            z_dim=100,
            beta1=0.5,
            critic_iterations=5,
            out_dir='extra',
            spec_norm=False,
            no_gp=False
    ):
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        self.out_dir = out_dir
        self.device = device
        self.num_gpu = num_gpu
        self.n_epochs = n_epochs
        self.dataloader: torch.utils.data.DataLoader = dataloader
        self.critic_iterations = critic_iterations
        self.crop_size = crop_size
        self.beta1 = beta1
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.z_dim = z_dim
        self.num_frames = num_frames
        self.spec_norm = spec_norm
        self.no_gp = no_gp
        if not spec_norm and no_gp:
            assert(0 == 1), 'Can\'t remove gradient penalty AND spectral normalization; Lipschitz-1 can\'t be enforced'

        self.step = 0
        self.epoch = 0
        self.checkpoint_file = os.path.join(self.out_dir, CHECKPOINT_FILENAME)

        if self.spec_norm:
            self.generator = self.get_generator()
            self.discriminator = self.get_discriminator()
        else:
            self.generator = self.get_generator()
            self.discriminator = self.get_discriminator()

        self._activation = {}
        self._register_hooks()

        self.discriminator_optimizer = self._make_optimizer(self.discriminator)
        self.generator_optimizer = self._make_optimizer(self.generator)
        self.to(device)

        self._experiment = experiment
        self._log_parameters()

    def _log_parameters(self):
        self._experiment.log_parameters({
            'start_epoch': self.epoch,
            'n_epochs': self.n_epochs,
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
        return torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.learning_rate,
            betas=(self.beta1, BETA2),
            weight_decay=self.weight_decay,
        )

    def to(self, device):
        """
        Moves generator and discriminator models to the specified device, updates model's device,
        and starts data parallelism

        Args:
            device: device to move tensors to

        Returns:
            None
        """
        self.device = device
        self.generator = torch.nn.DataParallel(self.generator, device_ids=range(self.num_gpu)).to(device)
        self.discriminator = torch.nn.DataParallel(self.discriminator, device_ids=range(self.num_gpu)).to(device)

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
        d_net = Discriminator(self.spec_norm, self.no_gp)
        d_net.apply(init_weights)
        return d_net

    def train(self):
        """
        Main training loop. Run this to train the models.
        """
        for self.epoch in range(self.epoch, self.epoch + self.n_epochs):
            self._experiment.set_epoch(self.epoch)
            for _, (batch, lbl) in enumerate(self.dataloader):
                self._increment_total_step()
                batch = batch.to(self.device)
                if (self.step + 1) % (self.critic_iterations + 1) != 0:
                    fake_batch = self.optimize(batch, DISCRIMINATOR)
                else:
                    fake_batch = self.optimize(batch, GENERATOR)
                self._interval_log(batch, fake_batch)
            self._experiment.log_epoch_end(self.epoch)

        self.save()
        self._save_batch_as_gif(batch, name=f'final-real', upload=True)
        self._save_batch_as_gif(fake_batch, name=f'final-fake', upload=True)
        self.log_model()

    def _interval_log(self, batch: torch.Tensor, fake_batch: torch.Tensor):
        """
        Perform logging tasks for the end of an epoch.

        Args:
            batch: batch of real data
            fake_batch: batch of generated data
        """
        if (self.step + 1) % FID_INTERVAL == 0:
            self._log_fid(batch, fake_batch)
        if (self.step + 1) % SAMPLE_INTERVAL == 0:
            self._save_batch_as_gif(fake_batch, name=f'{self.step:05d}-fake', upload=True)
        if (self.step + 1) % (SAMPLE_INTERVAL * 10) == 0:
            self._save_batch_as_gif(batch, name=f'{self.step:05d}-real', upload=True)
        if (self.step + 1) % SAVE_INTERVAL == 0:
            self.save()

    def _log_fid(self, real_batch: torch.Tensor, fake_batch: torch.Tensor) -> None:
        """
        Log the Frechet inception distance
        Choose conv5 because the dimensionality of conv4 is much too high to compute covariance efficiently.

        Args:
            real_batch: batch of real images
            fake_batch: batch of generated images

        Returns:
            None
        """
        batch_size = real_batch.shape[0]
        self.discriminator.requires_grad_(False)
        self.discriminator(real_batch)
        real_embedding = self._activation['d_conv5'].reshape(batch_size, -1)
        self.discriminator(fake_batch)
        fake_embedding = self._activation['d_conv5'].reshape(batch_size, -1)
        self.discriminator.requires_grad_(True)

        fid = frechet_inception_distance(fake_embedding, real_embedding)
        self._experiment.log_metric('fid', fid)

    def _register_hooks(self):
        """
        Register hooks on models. Should be called before `self.to` (before models get wrapped in DataParallel)
        """
        def get_activation(name):
            def hook(model, input, output):
                self._activation[name] = output.detach()

            return hook

        self.discriminator.conv5.register_forward_hook(get_activation('d_conv5'))

    def _save_batch_as_gif(self, batch, name='', upload=False):
        grid_length = 5
        directory = os.path.join(self.out_dir, 'samples')
        if not os.path.exists(directory):
            os.mkdir(directory)

        # produce 5x5 tiled jpegs from the first 25 videos in the batch. One video for each frame.
        temp_files = [os.path.join(directory, f'{name}_{i}.jpeg') for i in range(self.num_frames)]
        for i, out_file in enumerate(temp_files):
            frame_batch = batch[:grid_length ** 2, :, i, ...]  # select the ith frame from 25 videos
            frame_batch = frame_batch / 2 + 0.5  # [-1, 1] -> [0, 1]
            torchvision.utils.save_image(frame_batch, out_file, nrow=grid_length)

        # Convert jpegs to AVI and delete the images
        cmd = f"ffmpeg -y -f image2 -i {directory}/{name}_%d.jpeg {directory}/{name}.avi"
        print(cmd)
        os.system(cmd)
        for out_file in temp_files:
            os.remove(out_file)

        # convert AVI to GIF and delete the AVI
        cmd = f"ffmpeg -y -i {directory}/{name}.avi -pix_fmt rgb24 {directory}/{name}.gif"
        print(cmd)
        os.system(cmd)
        os.remove(f'{directory}/{name}.avi')

        # Upload to CometML
        if upload:
            self._experiment.log_image(f'{directory}/{name}.gif')

    def _increment_total_step(self):
        """
        Increment and log the total steps for Comet
        """
        self._experiment.set_step(self.step)
        self.step += 1

    def log_model(self):
        """
        Log the saved models to CometML
        """
        if not os.path.exists(self.checkpoint_file):
            raise FileNotFoundError(f'{self.checkpoint_file} file not found')
        self._experiment.log_model('GAN', self.checkpoint_file)

    def save(self):
        """
        Save model checkpoints
        """
        torch.save({
            'epoch': self.epoch,
            'step': self.step,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'generator_optimizer_state_dict': self.generator_optimizer.state_dict(),
            'discriminator_optimizer_state_dict': self.discriminator_optimizer.state_dict(),
        }, self.checkpoint_file)

    def load(self, checkpoint_path: str):
        """
        Load the optimizer and model state dictionaries from model_dir

        Args:
            checkpoint_path: filepath of the checkpoint

        Returns:
            None
        """
        assert os.path.exists(checkpoint_path), f'file {checkpoint_path} does not exist'
        checkpoint = torch.load(checkpoint_path)
        self.epoch = checkpoint['epoch']
        self.step = checkpoint.get('step', 0)  # use a default of 0 for backwards compatibility
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.generator_optimizer.load_state_dict(checkpoint['generator_optimizer_state_dict'])
        self.discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer_state_dict'])

    def optimize(self, batch, model_type):
        """
        Run optimization

        Args:
            batch: batch of real videos
            model_type: "generator" or "discriminator"

        Returns:
            batch of fake videos
        """
        assert model_type in {GENERATOR, DISCRIMINATOR}
        if model_type == GENERATOR:
            self.discriminator.requires_grad = False
            fake_videos = self._optimize_generator(batch.shape[0])
            self.discriminator.requires_grad = True
        elif model_type == DISCRIMINATOR:
            self.generator.requires_grad = False
            z_vec = torch.rand((batch.shape[0], self.z_dim), device=self.device)
            fake_videos = self.generator(z_vec)
            self._optimize_discriminator(batch, fake_videos.detach())
            self.generator.requires_grad = True
        return fake_videos

    def _optimize_discriminator(self, batch, fake_videos):

        self.discriminator.zero_grad()

        d_fake = self.discriminator(fake_videos)
        d_real = self.discriminator(batch)
        g_cost = -torch.mean(d_fake)
        d_cost = -g_cost - torch.mean(d_real)

        if (self.step + 1) % (self.critic_iterations + 1) == 1:
            self._experiment.log_metric('g_cost', g_cost)
        self._experiment.log_metric('d_fake', g_cost)
        self._experiment.log_metric('d_cost', d_cost)
        self._experiment.log_metric('d_real', torch.mean(d_real))

        # alpha = torch.rand(size=(batch.shape[0], 1, 1, 1, 1), device=self.device)
        # interpolates = batch + (alpha * (fake_videos - batch))
        # interpolates.requires_grad_(True)
        # interpolates.retain_grad()
        # d_hat = self.discriminator(interpolates)
        #
        # self.discriminator.requires_grad = False
        # d_hat.backward(torch.ones_like(d_hat), retain_graph=True)
        # self.discriminator.requires_grad = True
        #
        # gradients = interpolates.grad
        # slopes = torch.norm(gradients.reshape(len(batch), -1), p=2, dim=1)
        # gradient_penalty = torch.mean((slopes - 1.) ** 2)

        one = torch.FloatTensor([1])
        mone = one * -1
        one = one.to(self.device)
        mone = mone.to(self.device)
        d_real = torch.mean(d_real, dim=0)
        d_real.backward(mone)
        d_fake = torch.mean(d_fake, dim=0)
        d_fake.backward(one)

        # IDEA USE ZERO CENTERED GRADIENT PENALTY
        # IDEA USE DIFFERENT GRADIENT PENALTY CONSTANT
        # IDEA USE ADAM WEIGHT DECAY OF 0.001 (AS RECOMMENDED IN IMPROVED TRAINING OF WGAN PAPER)
        # IDEA USE REGULARIZING TERM FOR D LOSS TO PREVENT DRIFT FROM 0 (L* = L + cE[D(x)**2]), c=0.001
        # IDEA COMBINE GP WITH SN (REMOVE LN, as in http://proceedings.mlr.press/v97/kurach19a/kurach19a.pdf#page=1&zoom=100,0,0)

        if not self.no_gp:
            gradient_penalty = self._calc_grad_penalty(batch, fake_videos)
            gradient_penalty.backward()
            d_cost_final = d_cost + gradient_penalty
            self._experiment.log_metric('grad_penalty', gradient_penalty)
        else:
            d_cost_final = d_cost

        self.discriminator_optimizer.step()
        print(f'discriminator cost: {d_cost_final}')
        self._experiment.log_metric('d_cost_final', d_cost_final)
        # self.discriminator_optimizer.zero_grad()

    def _calc_grad_penalty(self, real_data, fake_data):
        alpha = torch.rand(real_data.size(0), 1, 1, 1, 1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.to(self.device)

        interpolates = alpha * real_data + ((1. - alpha) * fake_data)

        interpolates = interpolates.to(self.device)
        interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates = self.discriminator(interpolates)

        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(self.device),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * GRADIENT_MULTIPLIER
        return gradient_penalty

    def _optimize_generator(self, b_size):

        self.generator.zero_grad()

        z_vec = torch.rand((b_size, self.z_dim), device=self.device)
        fake_videos = self.generator(z_vec)

        d_fake = self.discriminator(fake_videos)
        g_cost = -torch.mean(d_fake)
        g_cost.backward()

        self.generator_optimizer.step()
        print(f'generator cost: {g_cost}')
        self._experiment.log_metric('d_fake', g_cost)
        # self.generator_optimizer.zero_grad()

        return fake_videos

import os

import torch
import torch.cuda
import torchvision.utils

from model.improved_video_gan import Generator, init_weights, Discriminator

CHECKPOINT_FILENAME = 'checkpoint.model'
SAVE_INTERVAL = 10
SAMPLE_INTERVAL = 1
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
            z_dim=100,
            beta1=0.5,
            critic_iterations=5,
            out_dir='extra'
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
        self.z_dim = z_dim
        self.num_frames = num_frames

        self.step = 0
        self.epoch = 0
        self.checkpoint_file = os.path.join(self.out_dir, CHECKPOINT_FILENAME)

        self.generator = self.get_generator()
        self.discriminator = self.get_discriminator()

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
        return torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.learning_rate,
            betas=(self.beta1, BETA2),
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
        d_net = Discriminator()
        d_net.apply(init_weights)
        return d_net

    def train(self):
        """
        Main training loop. Run this to train the models.
        """
        for epoch in range(self.epoch, self.epoch + self.n_epochs):
            self._experiment.set_epoch(epoch)
            for _, (batch, lbl) in enumerate(self.dataloader):
                self._increment_total_step()
                batch = batch.to(self.device)
                if (self.step + 1) % self.critic_iterations != 0:
                    fake_batch = self.optimize(batch, DISCRIMINATOR)
                else:
                    fake_batch = self.optimize(batch, GENERATOR)

            # Log and save checkpoint
            self._experiment.log_epoch_end(epoch)
            if (epoch + 1) % SAVE_INTERVAL == 0:
                self.save()
            if (epoch + 1) % SAMPLE_INTERVAL == 0:
                self._save_batch_as_gif(batch, name=f'{epoch:03d}-real', upload=True)
                self._save_batch_as_gif(fake_batch, name=f'{epoch:03d}-fake', upload=True)
        self.save()
        self._save_batch_as_gif(batch, name=f'final-real', upload=True)
        self._save_batch_as_gif(fake_batch, name=f'final-fake', upload=True)
        self.log_model()
        
    def _save_batch_as_gif(self, batch, name='', upload=False):
        grid_length = 5
        directory = os.path.join(self.out_dir, 'samples')
        if not os.path.exists(directory):
            os.mkdir(directory)

        # produce 5x5 tiled jpegs from the first 25 videos in the batch. One video for each frame.
        temp_files = [os.path.join(directory, f'{name}_{i}.jpeg') for i in range(self.num_frames)]
        for i, out_file in enumerate(temp_files):
            frame_batch = batch[:grid_length**2, :, i, ...]  # select the ith frame from 25 videos
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
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.generator_optimizer.load_state_dict(checkpoint['generator_optimizer_state_dict'])
        self.discriminator_optimizer.load_state_dict(checkpoint['discriminator_state_dict'])

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
        z_vec = torch.rand((batch.shape[0], self.z_dim), device=self.device)
        fake_videos = self.generator(z_vec)

        if model_type == GENERATOR:
            self.discriminator.requires_grad = False
            self._optimize_generator(fake_videos)
            self.discriminator.requires_grad = True
        elif model_type == DISCRIMINATOR:
            self.generator.requires_grad = False
            self._optimize_discriminator(batch, fake_videos.detach())
            self.generator.requires_grad = True
        return fake_videos

    def _optimize_discriminator(self, batch, fake_videos):

        self.discriminator.zero_grad()

        d_fake = self.discriminator(fake_videos)
        d_real = self.discriminator(batch)
        g_cost = -torch.mean(d_fake)
        d_cost = -g_cost - torch.mean(d_real)

        self._experiment.log_metric('g_cost', g_cost)
        self._experiment.log_metric('d_cost', d_cost)

        alpha = torch.rand(size=(batch.shape[0], 1, 1, 1, 1), device=self.device)
        interpolates = batch + (alpha * (fake_videos - batch))
        interpolates.requires_grad_(True)
        interpolates.retain_grad()
        d_hat = self.discriminator(interpolates)

        self.discriminator.requires_grad = False
        d_hat.backward(torch.ones_like(d_hat), retain_graph=True)
        self.discriminator.requires_grad = True

        gradients = interpolates.grad
        slopes = torch.sqrt(torch.sum(torch.square(gradients), dim=1))
        # noinspection PyTypeChecker
        gradient_penalty = torch.mean((slopes - 1.) ** 2)
        d_cost_final = d_cost + GRADIENT_MULTIPLIER * gradient_penalty

        d_cost_final.backward()
        self.discriminator_optimizer.step()
        print(f'discriminator cost: {d_cost_final}')
        self._experiment.log_metric('d_cost_final', d_cost_final)
        self.discriminator_optimizer.zero_grad()

    def _optimize_generator(self, fake_videos):

        self.generator.zero_grad()

        d_fake = self.discriminator(fake_videos)
        g_cost = -torch.mean(d_fake)
        g_cost.backward()

        self.generator_optimizer.step()
        print(f'generator cost: {g_cost}')
        self._experiment.log_metric('g_cost', g_cost)
        self.generator_optimizer.zero_grad()
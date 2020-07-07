import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.utils as tu


class generator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, input_dim=100, output_dim=1, input_size=32):
        super(generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.BatchNorm1d(128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.LeakyReLU(0.2),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Tanh(),
        )
        self.reset_parameters()

    def forward(self, input):
        x = self.fc(input)
        x = x.view(-1, 128, (self.input_size // 4), (self.input_size // 4))
        x = self.deconv(x)

        return x

    def reset_parameters(self):
        """Ported from utils."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            #  elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            #      m.weight.data.fill_(1.)
            #      m.bias.data.zero_()


class discriminator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, input_dim=1, output_dim=1, input_size=32):
        super(discriminator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * (self.input_size // 4) * (self.input_size // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.output_dim),
        )
        self.reset_parameters()

    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 128 * (self.input_size // 4) * (self.input_size // 4))
        x = self.fc(x)

        return x

    def reset_parameters(self):
        """Ported from utils."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            #  elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            #      m.weight.data.fill_(1.)
            #      m.bias.data.zero_()


class GAN(object):
    def __init__(self, in_channels, input_size,
                 rand_type='rand', z_dim=62,
                 epoch=50, batch_size=64,
                 lr_g=0.0002, lr_d=0.0002, beta1=0.5, beta2=0.999, disc_iters=1,
                 gpu_to_use=0, writer=None):
        # parameters
        self.model_name = type(self).__name__
        self.epoch = epoch
        self.batch_size = batch_size
        if gpu_to_use is not None and torch.cuda.is_available():
            self.device = torch.device('cuda:{}'.format(gpu_to_use))
        else:
            self.device = torch.device('cpu')

        # NOTE: the data_loader layout follows pytorch conventions, i.e. the
        # data layout is NCHW.
        self.z_dim = z_dim
        self.rand_type = rand_type

        # network init
        self.G = generator(input_dim=self.z_dim, output_dim=in_channels,
                           input_size=input_size).to(self.device)
        self.D = discriminator(input_dim=in_channels, output_dim=1,
                               input_size=input_size).to(self.device)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=lr_g, betas=(beta1, beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=lr_d, betas=(beta1, beta2))
        self.BCE_loss = nn.BCEWithLogitsLoss().to(self.device)

        self.sample_z_ = self.prior_generator()
        self.disc_iters = disc_iters
        # If not None, use SummaryWriter from tensorboardX
        self.writer = writer

    def prior_generator(self):
        """Generate prior distribution for G."""
        if self.rand_type == 'rand':
            return torch.rand(self.batch_size, self.z_dim, device=self.device).mul(2).sub(1)  # [-1,1]
        elif self.rand_type == 'randn':
            return torch.randn(self.batch_size, self.z_dim, device=self.device)
        else:
            raise KeyError('Unknown rand type: {}'.format(self.rand_type))

    def fit(self, x_train, callback=None):
        """x_train is a torch tensor with layout NCHW and range [-1, 1]."""

        y_real_ = torch.ones(self.batch_size, 1, device=self.device)
        y_fake_ = torch.zeros(self.batch_size, 1, device=self.device)

        self.D.train()
        print('training start!!')
        tic = time.time()
        self.global_step = 0
        for epoch in range(self.epoch):
            self.G.train()
            data_loader = data.DataLoader(
                data.TensorDataset(x_train),
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=True,  # as implemented in the original version
            )
            for it, (x_, ) in enumerate(data_loader):
                x_ = x_.to(self.device)
                z_ = self.prior_generator()

                # update D network
                self.D_optimizer.zero_grad()

                D_real = self.D(x_)
                D_real_loss = self.BCE_loss(D_real, y_real_)

                G_ = self.G(z_)
                D_fake = self.D(G_)
                D_fake_loss = self.BCE_loss(D_fake, y_fake_)

                D_loss = D_real_loss + D_fake_loss
                if self.writer is not None:
                    # We record the true value of log-likelihood, i.e.
                    # -y_i*log(D(x_i)) + -(1-y_i)*log(1-D(G(z_i))).
                    # In this case, the loss will decrease instead of increase.
                    self.writer.add_scalar('D_loss', D_loss.item(), self.global_step)

                D_loss.backward()
                self.D_optimizer.step()

                if (self.global_step + 1) % self.disc_iters == 0:

                    # update G network
                    self.G_optimizer.zero_grad()

                    G_ = self.G(z_)
                    D_fake = self.D(G_)
                    G_loss = self.BCE_loss(D_fake, y_real_)
                    if self.writer is not None:
                        # The logged loss is
                        # -y_i*log(D(G(z_i)))
                        self.writer.add_scalar('G_loss', G_loss.item(), self.global_step)

                    G_loss.backward()
                    self.G_optimizer.step()

                    if self.writer is not None and (self.global_step+1) % 100 == 0:
                        # record image every 100 iterations
                        self.G.eval()
                        generated = self.G(self.sample_z_)
                        grid = tu.make_grid(generated, nrow=int(np.sqrt(generated.size(0))),
                                            normalize=True, range=(-1, 1))
                        self.writer.add_image(self.rand_type, grid, self.global_step)
                        self.G.train()

                    if (time.time()-tic) > 5:  # print every 5 sec
                        print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
                              ((epoch + 1), (it + 1), x_train.size(0)//self.batch_size, D_loss.item(), G_loss.item()))
                        tic = time.time()
                self.global_step += 1

                if callable(callback) and callback(self):
                    break

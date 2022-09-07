from torch import nn
import torch

class Sine(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)


class AdaIN(nn.Module):
    def __init__(self, channels, latent_size):
        super().__init__()
        self.sine_fc = nn.Sequential(nn.Linear(1, latent_size), Sine())
        self.channels = channels
        self.linear = nn.Sequential(nn.Linear(latent_size, (channels+latent_size)//2),
                                    nn.ELU(),
                                    nn.Linear((channels+latent_size)//2, channels*2))

    def forward(self, x, t):
        dlatent = self.sine_fc(t)
        x = nn.InstanceNorm2d(self.channels)(x)
        style = self.linear(dlatent)
        style = style.view([-1, 2, x.size()[1]] + [1] * (len(x.size()) - 2))
        return x * (style[:, 0] + 1) + style[:, 1]


class ResBlock(nn.Module):
    def __init__(self, filters, subsample=False):
        super().__init__()
        """
        A 2-layer residual learning building block as illustrated by Fig.2
        in "Deep Residual Learning for Image Recognition"

        Parameters:

        - filters:   int
                     the number of filters for all layers in this block

        - subsample: boolean
                     whether to subsample the input feature maps with stride 2
                     and doubling in number of filters

        Attributes:

        - shortcuts: boolean
                     When false the residual shortcut is removed
                     resulting in a 'plain' convolutional block.
        """
        # Determine subsampling
        s = 0.5 if subsample else 1.0

        # Setup layers
        self.conv1 = nn.Conv2d(int(filters * s), filters, kernel_size=3,
                               stride=int(1 / s), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filters, track_running_stats=True)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(filters, track_running_stats=True)
        self.relu2 = nn.ReLU()

        # Shortcut downsampling
        self.downsample = nn.AvgPool2d(kernel_size=1, stride=2)

        # Initialise weights according to the method described in
        # “Delving deep into rectifiers: Surpassing human-level performance on ImageNet
        # classification” - He, K. et al. (2015)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def shortcut(self, z, x):
        """
        Implements parameter free shortcut connection by identity mapping.
        If dimensions of input x are greater than activations then this
        is rectified by downsampling and then zero padding dimension 1
        as described by option A in paper.

        Parameters:
        - x: tensor
             the input to the block
        - z: tensor
             activations of block prior to final non-linearity
        """
        if x.shape != z.shape:
            d = self.downsample(x)
            p = torch.mul(d, 0)
            return z + torch.cat((d, p), dim=1)
        else:
            return z + x

    def forward(self, x, shortcuts=False):
        z = self.conv1(x)
        z = self.bn1(z)
        z = self.relu1(z)

        z = self.conv2(z)
        z = self.bn2(z)

        # Shortcut connection
        # This if statement is the only difference between
        # a convolutional net and a resnet!
        if shortcuts:
            z = self.shortcut(z, x)

        z = self.relu2(z)

        return z

class DownBlock(nn.Sequential):

    def __init__(self, in_ch, out_ch):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class UpBlock(nn.Sequential):

    def __init__(self, in_ch, out_ch):
        super().__init__(
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )


class SimpleAE(nn.Module):
    def __init__(self, in_channels, filters, n_res_blocks=4, d_latent=64):
        super().__init__()
        self.n_res_blocks = n_res_blocks
        self.d_latent = d_latent
        self.down = nn.Sequential(
            DownBlock(in_channels, filters),
            DownBlock(filters, filters*2))
        self.residuals = nn.ModuleList([ResBlock(filters*2) for _ in range(self.n_res_blocks)])
        self.up = nn.Sequential(
            UpBlock(filters*2, filters),
            UpBlock(filters, filters),
            nn.Conv2d(filters, in_channels, 3, 1, 1),
            )
        self.adains = nn.ModuleList([AdaIN(filters*2, self.d_latent) for _ in range(self.n_res_blocks)])

    def forward(self, img, t):
        emb = self.down(img)
        for i in range(self.n_res_blocks):
            emb = self.residuals[i](emb)
            emb = self.adains[i](emb, t)
        pred_img = self.up(emb)
        return pred_img


if __name__ == "__main__":
    ae = SimpleAE(1, 32)
    img = torch.randn(2, 1, 28, 28)
    t = torch.randn(2, 1)
    pred_img = ae(img, t)
    print(pred_img.shape)

import torch.nn as nn


#Weight initialization
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, in_channel=100,
                 out_channel=3,
                 feature_map=128,
                 kernel_size=4,
                 image_size=64,
                 ngpu=0):
        super(Generator, self).__init__()
        self.ngpu = ngpu

        # Upsample random variable
        layers = nn.ModuleList()
        layers += [nn.ConvTranspose2d(in_channel, feature_map, 4, 1, 0)]
        layers += [nn.BatchNorm2d(feature_map)]
        layers += [nn.ReLU(True)]

        size = 4

        layers += [nn.ConvTranspose2d(feature_map, feature_map // 2, kernel_size, 2, 1, bias=False)]
        layers += [nn.BatchNorm2d(feature_map // 2)]
        layers += [nn.ReLU(True)]
        feature_map = feature_map // 2
        size = size * 2

        # Main G structure
        while size < image_size // 2:
            layers += [nn.ConvTranspose2d(feature_map, feature_map // 2, 4, 2, 1, bias=False)]
            layers += [nn.BatchNorm2d(feature_map // 2)]
            layers += [nn.ReLU(True)]
            feature_map = feature_map // 2
            size = size * 2

        # Final layer
        layers += [nn.ConvTranspose2d(feature_map, out_channel, 4, 2, 1, bias=False)]
        layers += [nn.Tanh()]

        self.g = nn.Sequential(*layers)

    def forward(self, z):
        return self.g(z)


class Discriminator(nn.Module):
    def __init__(self, in_channel=3,
                 out_channel=1,
                 feature_map=32,
                 kernel_size=4,
                 image_size=64,
                 ngpu = 0,
                 dcgan = False):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.dcgan = dcgan
        # First layer
        layers = nn.ModuleList()
        layers += [nn.Conv2d(in_channel, feature_map, 4, 2, 1)]
        layers += [nn.BatchNorm2d(feature_map)]
        layers += [nn.LeakyReLU(0.2, inplace=True)]

        size = image_size / 2

        # Main D structure
        while size > 8:
            layers += [nn.Conv2d(feature_map, feature_map * 2, 4, 2, 1, bias=False)]
            layers += [nn.BatchNorm2d(feature_map * 2)]
            layers += [nn.LeakyReLU(0.2, inplace=True)]
            feature_map = feature_map * 2
            size = size / 2

        layers += [nn.Conv2d(feature_map, feature_map * 2, kernel_size, 2, 1, bias=False)]
        layers += [nn.BatchNorm2d(feature_map * 2)]
        layers += [nn.LeakyReLU(0.2, inplace=True)]
        feature_map = feature_map * 2

        # Final layer
        layers += [nn.Conv2d(feature_map, out_channel, 4, 1, 0, bias=False)]
        if self.dcgan:
            layers += [nn.Sigmoid()]

        self.d = nn.Sequential(*layers)

    def forward(self, image):
        return self.d(image)

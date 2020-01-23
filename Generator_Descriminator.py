import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, batch_size):
        super(Generator, self).__init__()

        # Свертка [3x256x256] -> [64x128x128]
        self.conv1 = nn.Conv2d(3, 64, 4, 2, 1)

        # Свертка -> [128x64x64]
        self.conv2 = nn.Sequential(
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Conv2d(64, 128, 4, 2, 1),
                                   nn.BatchNorm2d(128)
                                   )

        # Свертка -> [256x32x32]
        self.conv3 = nn.Sequential(
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Conv2d(128, 256, 4, 2, 1),
                                   nn.BatchNorm2d(256)
                                   )

        # Свертка -> [512x16x16]
        self.conv4 = nn.Sequential(
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Conv2d(256, 512, 4, 2, 1),
                                   nn.BatchNorm2d(512)
                                   )

        # Свертка -> [512x8x8]
        self.conv5 = nn.Sequential(
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Conv2d(512, 512, 4, 2, 1),
                                   nn.BatchNorm2d(512)
                                   )

        # Свертка -> [512x4x4]
        self.conv6 = nn.Sequential(
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Conv2d(512, 512, 4, 2, 1),
                                   nn.BatchNorm2d(512)
                                   )

        # Свертка -> [512x2x2]
        self.conv7 = nn.Sequential(
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Conv2d(512, 512, 4, 2, 1),
                                   nn.BatchNorm2d(512)
                                   )

        # Свертка -> [512x1x1]
        self.conv8 = nn.Sequential(
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Conv2d(512, 512, 4, 2, 1),
                                   nn.BatchNorm2d(512)
                                   )

        # Deconvolution -> [512x2x2]
        self.deconv8 = nn.Sequential(
                                     nn.ReLU(),
                                     nn.ConvTranspose2d(512, 512, 4, 2, 1),
                                     nn.BatchNorm2d(512), nn.Dropout(0.5)
                                     )

        # Deconvolution [(512+512)x2x2] -> [512x4x4]
        self.deconv7 = nn.Sequential(
                                     nn.ReLU(),
                                     nn.ConvTranspose2d(512 * 2, 512, 4, 2, 1),
                                     nn.BatchNorm2d(512), nn.Dropout(0.5)
                                     )

        # Deconvolution [(512+512)x4x4] -> [512x8x8]
        self.deconv6 = nn.Sequential(
                                     nn.ReLU(),
                                     nn.ConvTranspose2d(512 * 2, 512, 4, 2, 1),
                                     nn.BatchNorm2d(512), nn.Dropout(0.5)
                                     )

        # Deconvolution [(512+512)x8x8] -> [512x16x16]
        self.deconv5 = nn.Sequential(
                                     nn.ReLU(),
                                     nn.ConvTranspose2d(512 * 2, 512, 4, 2, 1),
                                     nn.BatchNorm2d(512)
                                     )

        # Deconvolution [(512+512)x16x16] -> [256x32x32]
        self.deconv4 = nn.Sequential(
                                     nn.ReLU(),
                                     nn.ConvTranspose2d(512 * 2, 256, 4, 2, 1),
                                     nn.BatchNorm2d(256)
                                     )

        # Deconvolution [(256+256)x32x32] -> [128x64x64]
        self.deconv3 = nn.Sequential(
                                     nn.ReLU(),
                                     nn.ConvTranspose2d(256 * 2, 128, 4, 2, 1),
                                     nn.BatchNorm2d(128)
                                     )

        # Deconvolution [(128+128)x64x64] -> [64x128x128]
        self.deconv2 = nn.Sequential(
                                     nn.ReLU(),
                                     nn.ConvTranspose2d(128 * 2, 64, 4, 2, 1),
                                     nn.BatchNorm2d(64)
                                     )

        # Deconvolution [(64+64)x128x128] -> [3x256x256]
        self.deconv1 = nn.Sequential(
                                     nn.ReLU(),
                                     nn.ConvTranspose2d(64 * 2, 3, 4, 2, 1),
                                     nn.Tanh()
                                     )

    # Прямое распространение ошибки
    def forward(self, x):

        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)
        c6 = self.conv6(c5)
        c7 = self.conv7(c6)
        c8 = self.conv8(c7)

        d7 = self.deconv8(c8)
        d7 = torch.cat((c7, d7), dim=1)
        d6 = self.deconv7(d7)
        d6 = torch.cat((c6, d6), dim=1)
        d5 = self.deconv6(d6)
        d5 = torch.cat((c5, d5), dim=1)
        d4 = self.deconv5(d5)
        d4 = torch.cat((c4, d4), dim=1)
        d3 = self.deconv4(d4)
        d3 = torch.cat((c3, d3), dim=1)
        d2 = self.deconv3(d3)
        d2 = torch.cat((c2, d2), dim=1)
        d1 = self.deconv2(d2)
        d1 = torch.cat((c1, d1), dim=1)
        out = self.deconv1(d1)

        return out

class Discriminator(nn.Module):
    def __init__(self, batch_size):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
                                  nn.Conv2d(3*2, 64, 4, 2, 1), # Свертка [(3+3)x256x256] -> [64x128x128] -> [128x64x64]
                                  nn.LeakyReLU(0.2, inplace=True),
                                  nn.Conv2d(64, 128, 4, 2, 1),
                                  nn.BatchNorm2d(128),
                                  nn.LeakyReLU(0.2, inplace=True), # Свертка -> [256x32x32]
                                  nn.Conv2d(128, 256, 4, 2, 1),
                                  nn.BatchNorm2d(256),
                                  nn.LeakyReLU(0.2, inplace=True), # Свертка -> [512x31x31]
                                  nn.Conv2d(256, 512, 4, 1, 1),
                                  nn.BatchNorm2d(512),
                                  nn.LeakyReLU(0.2, inplace=True), # Свертка -> [1x30x30] (PatchGAN)
                                  nn.Conv2d(512, 1, 4, 1, 1),
                                  nn.Sigmoid()
                                  )

    def forward(self, x1, x2): # Для настоящего и для фейкового изображения
        out = torch.cat((x1, x2), dim=1)
        return self.main(out)
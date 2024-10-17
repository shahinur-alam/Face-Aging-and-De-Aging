import torch
import torch.nn as nn


class STGAN(nn.Module):
    def __init__(self):
        super(STGAN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256 + 13, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x, attr):
        encoded = self.encoder(x)
        b, c, h, w = encoded.size()
        attr = attr.view(b, 13, 1, 1).expand(b, 13, h, w)
        latent = torch.cat([encoded, attr], dim=1)
        return self.decoder(latent)


def load_pretrained_stgan(path):
    model = STGAN()
    model.load_state_dict(torch.load(path))
    return model
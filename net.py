import torch
from torch._C import ModuleDict
import torch.nn as nn
from torch.nn.utils import weight_norm

from vector_quantize_pytorch import ResidualVQ

# Generator


class ResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()

        self.layers = nn.Sequential([
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=7, dilation=dilation),
            nn.ELU(),
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=1)
        ])

    def forward(self, x):
        return x + self.layers(x)


class EncoderBlock(nn.Module):
    def __init__(self, out_channels, stride):
        super().__init__()

        self.layers = nn.Sequential([
            ResidualUnit(in_channels=out_channels//2,
                         out_channels=out_channels//2, dilation=1),
            nn.ELU(),
            ResidualUnit(in_channels=out_channels//2,
                         out_channels=out_channels//2, dilation=3),
            nn.ELU(),
            ResidualUnit(in_channels=out_channels//2,
                         out_channels=out_channels//2, dilation=9),
            nn.ELU(),
            nn.Conv1d(in_channels=out_channels//2, out_channels=out_channels,
                      kernel_size=2*stride, stride=stride)
        ])

    def forward(self, x):
        return self.layers(x)


class DecoderBlock(nn.Module):
    def __init__(self, out_channels, stride):
        super().__init__()

        self.layers = nn.Sequential([
            nn.ConvTranspose1d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=2*stride, stride=stride),
            nn.ELU(),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels,
                         dilation=1),
            nn.ELU(),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels,
                         dilation=3),
            nn.ELU(),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels,
                         dilation=9),

        ])

    def forward(self, x):
        return self.layers(x)


class Encoder(nn.Module):
    def __init__(self, C, D):
        super().__init__()

        self.layers = nn.Sequential([
            nn.Conv1d(in_channels=1, out_channels=C, kernel_size=7),
            nn.ELU(),
            EncoderBlock(out_channels=2*C, stride=2),
            nn.ELU(),
            EncoderBlock(out_channels=4*C, stride=4),
            nn.ELU(),
            EncoderBlock(out_channels=8*C, stride=5),
            nn.ELU(),
            EncoderBlock(out_channels=16*C, stride=8),
            nn.ELU(),
            nn.Conv1d(in_channels=16*C, out_channels=D, kernel_size=3)
        ])

    def forward(self, x):
        return self.layers(x)


class Decoder(nn.Module):
    def __init__(self, C, D):
        super().__init__()
        
        self.layers = nn.Sequential([
            nn.Conv1d(in_channels=D, out_channels=16*C, kernel_size=7),
            nn.ELU(),
            DecoderBlock(out_channels=8*C, stride=8),
            nn.ELU(),
            DecoderBlock(out_channels=4*C, stride=5),
            nn.ELU(),
            DecoderBlock(out_channels=2*C, stride=4),
            nn.ELU(),
            DecoderBlock(out_channels=C, stride=2),
            nn.ELU(),
            nn.Conv1d(in_channels=C, out_channels=1, kernel_size=7)
        ])
    
    def forward(self, x):
        return self.layers(x)


class SoundStream(nn.Module):
    def __init__(self, C, D, n_q, codebook_size):
        super().__init__()

        self.encoder = Encoder(C=C, D=D)
        self.quantizer = ResidualVQ(
            num_quantizers=n_q, dim=D, codebook_size=codebook_size,
            kmeans_init=True, kmeans_iters=100, threshold_ema_dead_code=2
        )
        self.decoder = Decoder(C=C, D=D)
    
    def forward(self, x):
        e = self.encoder(x)
        quantized, _ = self.quantizer(e)
        o = self.decoder(quantized)
        return o

# Wave-based Discriminator


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


class WaveDiscriminatorBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.ModuleList([
            nn.Sequential([
                nn.ReflectionPad1d(7),
                WNConv1d(in_channels=1, out_channels=16, kernel_size=15),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            ]),
            nn.Sequential([
                WNConv1d(in_channels=16, out_channels=64, kernel_size=41,
                         stride=4, padding=20, groups=4),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            ]),
            nn.Sequential([
                WNConv1d(in_channels=64, out_channels=256, kernel_size=41,
                         stride=4, padding=20, groups=16),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            ]),
            nn.Sequential([
                WNConv1d(in_channels=256, out_channels=1024, kernel_size=41,
                         stride=4, padding=20, groups=64),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            ]),
            nn.Sequential([
                WNConv1d(in_channels=1024, out_channels=1024, kernel_size=41,
                         stride=4, padding=20, groups=256),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            ]),
            nn.Sequential([
                WNConv1d(in_channels=1024, out_channels=1024, kernel_size=5,
                         stride=1, padding=2),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            ]),
            WNConv1d(in_channels=1024, out_channels=1, kernel_size=3, stride=1,
                     padding=1)
        ])

    def forward(self, x):
        feature_map = []
        for layer in self.layers:
            x = layer(x)
            feature_map.append(x)
        return feature_map


class WaveDiscriminator(nn.Module):
    def __init__(self, num_D, downsampling_factor):
        super().__init__()
        
        self.num_D = num_D
        self.downsampling_factor = downsampling_factor
        
        self.model = ModuleDict({
            f"disc_{downsampling_factor**i}": WaveDiscriminatorBlock()
            for i in range(num_D)
        })
        self.downsampler = nn.AvgPool1d(kernel_size=4, stride=2, padding=1,
                                        count_include_pad=False)
    
    def forward(self, x):
        results = []
        for i in range(self.num_D):
            disc = self.model[f"disc_{self.downsampling_factor**i}"]
            results.append(disc(x))
            x = self.downsampler(x)
        return results


# STFT-based Discriminator

class ResidualUnit2d(nn.Module):
    def __init__(self, in_channels, N, m, s_t, s_f):
        super().__init__()

        self.layers = nn.Sequential([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=N,
                kernel_size=(3, 3)
            ),
            nn.ELU(),
            nn.Conv2d(
                in_channels=N,
                out_channels=m*N,
                kernel_size=(s_t+2, s_f+2),
                stride=(s_t, s_f)
            )
        ])

    def forward(self, x):
        return x + self.layers(x)


class STFTDiscriminator(nn.Module):
    def __init__(self, C, F):
        super().__init__()

        self.layers = nn.Sequential([
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(7, 7)),
            nn.ELU(),
            ResidualUnit2d(in_channels=32,  N=C,   m=2, s_t=1, s_f=2),
            nn.ELU(),
            ResidualUnit2d(in_channels=2*C, N=2*C, m=2, s_t=2, s_f=2),
            nn.ELU(),
            ResidualUnit2d(in_channels=4*C, N=4*C, m=1, s_t=1, s_f=2),
            nn.ELU(),
            ResidualUnit2d(in_channels=4*C, N=4*C, m=2, s_t=2, s_f=2),
            nn.ELU(),
            ResidualUnit2d(in_channels=8*C, N=8*C, m=1, s_t=1, s_f=2),
            nn.ELU(),
            ResidualUnit2d(in_channels=32,  N=8*C, m=2, s_t=2, s_f=2),
            nn.ELU(),
            nn.Conv2d(in_channels=16*C, out_channels=1,
                      kernel_size=(1, F/2**6))
        ])

    def forward(self, x):
        return self.layers(x)

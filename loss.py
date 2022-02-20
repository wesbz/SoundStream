import torch
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram


def adversarial_g_loss(features_stft_disc_G_x, features_wave_disc_G_x, lengths_stft, lengths_wave):
    wave_disc_names = lengths_wave.keys()
    
    stft_loss = F.relu(1-features_stft_disc_G_x[-1]).sum(dim=3).squeeze()/lengths_stft[-1].squeeze()
    wave_loss = torch.cat([F.relu(1-features_wave_disc_G_x[key][-1]).sum(dim=2).squeeze()/lengths_wave[key][-1].squeeze() for key in wave_disc_names])
    loss = torch.cat([stft_loss, wave_loss]).mean()
    
    return loss

def feature_loss(features_stft_disc_x, features_wave_disc_x, features_stft_disc_G_x, features_wave_disc_G_x, lengths_wave, lengths_stft):
    wave_disc_names = lengths_wave.keys()
    
    stft_loss = torch.stack([((feat_x-feat_G_x).abs().sum(dim=-1)/lengths_stft[i].view(-1,1,1)).sum(dim=-1).sum(dim=-1) for i, (feat_x, feat_G_x) in enumerate(zip(features_stft_disc_x, features_stft_disc_G_x))], dim=1).mean(dim=1, keepdim=True)
    wave_loss = torch.stack([torch.stack([(feat_x-feat_G_x).abs().sum(dim=-1).sum(dim=-1)/lengths_wave[key][i] for i, (feat_x, feat_G_x) in enumerate(zip(features_wave_disc_x[key], features_wave_disc_G_x[key]))], dim=1) for key in wave_disc_names], dim=2).mean(dim=1)
    loss = torch.cat([stft_loss, wave_loss], dim=1).mean()
    
    return loss


def spectral_reconstruction_loss(x, G_x, sr, dev, eps=1e-4):
    """
    Device must be specified because the window function and spctrogram must be on the same device
    """
    L = 0
    for i in range(6,12):
        s = 2**i
        alpha_s = (s/2)**0.5
        melspec = MelSpectrogram(sample_rate=sr, n_fft=s, hop_length=s//4, n_mels=8, wkwargs={"device": dev}).to(dev)
        S_x = melspec(x)
        S_G_x = melspec(G_x)
        
        loss = (S_x-S_G_x).abs().sum() + alpha_s*(((torch.log(S_x.abs()+eps)-torch.log(S_G_x.abs()+eps))**2).sum(dim=-2)**0.5).sum()
        L += loss
    
    return L


def adversarial_d_loss(features_stft_disc_x, features_wave_disc_x, features_stft_disc_G_x, features_wave_disc_G_x, lengths_stft, lengths_wave):
    wave_disc_names = lengths_wave.keys()
    
    real_stft_loss = F.relu(1-features_stft_disc_x[-1]).sum(dim=3).squeeze()/lengths_stft[-1].squeeze()
    real_wave_loss = torch.stack([F.relu(1-features_wave_disc_x[key][-1]).sum(dim=-1).squeeze()/lengths_wave[key][-1].squeeze() for key in wave_disc_names], dim=1)
    real_loss = torch.cat([real_stft_loss.view(-1,1), real_wave_loss], dim=1).mean()
    
    generated_stft_loss = F.relu(1+features_stft_disc_G_x[-1]).sum(dim=-1).squeeze()/lengths_stft[-1].squeeze()
    generated_wave_loss = torch.stack([F.relu(1+features_wave_disc_G_x[key][-1]).sum(dim=-1).squeeze()/lengths_wave[key][-1].squeeze() for key in wave_disc_names], dim=1)
    generated_loss = torch.cat([generated_stft_loss.view(-1,1), generated_wave_loss], dim=1).mean()
    
    return real_loss + generated_loss

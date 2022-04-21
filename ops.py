import contextlib
import torch

from loss import criterion_d, criterion_g

def transform(x, device):
    return torch.stft(x.squeeze(), n_fft=1024, hop_length=256, window=torch.hann_window(window_length=1024, device=device), return_complex=False).permute(0, 3, 1, 2)


def evaluate(generator, stft_disc, wave_disc, stft, x_loader, optimizer_g, optimizer_d, device, sr, train=False, history=None):
    models = [generator, stft_disc, wave_disc]
    with contextlib.ExitStack() as s:
        if not train:
            s.enter_stack(torch.no_grad())
            map(lambda x: x.eval(), models)
        else:
            map(lambda x: x.train(), models)

        tot_loss_d = 0.
        tot_loss_g = 0.

        for x, lengths_x in tqdm(x_loader):
            x = x.to(device)
            lengths_x = lengths_x.to(device)
            # evaluate generator
            G_x = generator(x)
            s_x = stft(x, device)

            lengths_s_x = 1 + torch.div(lengths_x, 256, rounding_mode="floor")

            s_G_x = stft(G_x, device)
            
            lengths_stft = stft_disc.features_lengths(lengths_s_x)
            lengths_wave = wave_disc.features_lengths(lengths_x)
            
            features_stft_disc_x = stft_disc(s_x)
            features_wave_disc_x = wave_disc(x)
            
            features_stft_disc_G_x = stft_disc(s_G_x)
            features_wave_disc_G_x = wave_disc(G_x)
            
            loss_g = criterion_g(x, G_x, features_stft_disc_x, features_wave_disc_x, features_stft_disc_G_x, features_wave_disc_G_x, lengths_wave, lengths_stft)
            tot_loss_g += loss_g.item()
            
            if training:
                # train generator
                optimizer_g.zero_grad()
                loss_g.backward()
                optimizer_g.step()
            
            # train discriminator
            features_stft_disc_x = stft_disc(s_x)
            features_wave_disc_x = wave_disc(x)

            features_stft_disc_G_x_det = stft_disc(s_G_x.detach())  # detach so generator isnt changed
            features_wave_disc_G_x_det = wave_disc(G_x.detach())  # detach so generator isnt changed
            
            loss_d = criterion_d(features_stft_disc_x, features_wave_disc_x, features_stft_disc_G_x_det, features_wave_disc_G_x_det, lengths_stft, lengths_wave)
            
            tot_loss_d += loss_d.item()
            
            optimizer_d.zero_grad()
            loss_d.backward()
            optimizer_d.step()

            if history:
                history["d"].append(tot_loss_d/len(x_loader))
                history["g"].append(tot_loss_g/len(x_loader))

    return generator, stft_disc, wave_disc, history

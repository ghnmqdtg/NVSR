# import git
# git_root = git.Repo("", search_parent_directories=True).git.rev_parse("--show-toplevel")
# import sys
# sys.path.append(git_root)

import librosa
import torch
import torch.nn as nn
import torchaudio
import numpy as np
from skimage.metrics import structural_similarity as ssim
from ssr_eval.utils import *

EPS = 1e-12


class AudioMetrics:
    def __init__(self, rate):
        self.rate = rate
        self.hop_length = int(rate / 100)
        self.n_fft = int(2048 / (44100 / rate))
        self.stft = STFTMag(nfft=self.n_fft, hop=self.hop_length)

    def read(self, est, target):
        # Use librosa to read the file
        # est, _ = librosa.load(est, sr=self.rate, mono=True)
        # target, _ = librosa.load(target, sr=self.rate, mono=True)

        # Use torchaudio to read the file
        est, _ = torchaudio.load(est)
        target, _ = torchaudio.load(target)

        return est, target

    def wav_to_spectrogram(self, wav):
        f = np.abs(librosa.stft(wav, hop_length=self.hop_length, n_fft=self.n_fft))
        f = np.transpose(f, (1, 0))
        # f.shape = torch.Size([1, 1, time, freq]), where freq == (1 + n_fft/2)
        f = torch.tensor(f[None, None, ...])
        return f

    def center_crop(self, x, y):
        dim = 2
        if x.size(dim) == y.size(dim):
            return x, y
        elif x.size(dim) > y.size(dim):
            offset = x.size(dim) - y.size(dim)
            start = offset // 2
            end = offset - start
            x = x[:, :, start:-end, :]
        elif x.size(dim) < y.size(dim):
            offset = y.size(dim) - x.size(dim)
            start = offset // 2
            end = offset - start
            y = y[:, :, start:-end, :]
        assert (
            offset < 10
        ), "Error: the offset %s is too large, check the code please" % (offset)
        return x, y

    def evaluation(self, est, target, file, cutoff_freq, device="cuda"):
        """evaluate between two audio
        Args:
            est (str or np.array): _description_
            target (str or np.array): _description_
            file (str): _description_
            cutoff_freq (int): The cutoff frequency for lowpass filter
            device (str, optional): Defaults to "cuda".

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        # import time; start = time.time()
        if type(est) != type(target):
            raise ValueError(
                "The input value should either both be numpy array or strings"
            )
        if type(est) == type(""):
            est_wav, target_wav = self.read(est, target)
            est_wav = est_wav.to(device)
            target_wav = est_wav.to(device)
        else:
            assert len(list(est.shape)) == 1 and len(list(target.shape)) == 1, (
                "The input numpy array shape should be [samples,]. Got input shape %s and %s. "
                % (est.shape, target.shape)
            )
            est_wav, target_wav = est, target
            # Convert to torch tensor
            est_wav = torch.tensor(est_wav).to(device)
            target_wav = torch.tensor(target_wav).to(device)

        assert (
            abs(target_wav.shape[0] - est_wav.shape[0]) < 100
        ), "Error: Shape mismatch between target and estimation %s and %s" % (
            str(target_wav.shape),
            str(est_wav.shape),
        )

        min_len = min(target_wav.shape[0], est_wav.shape[0])
        target_wav, est_wav = target_wav[:min_len], est_wav[:min_len]

        result = {}

        # Highcut frequency = int((1 + n_fft/2)
        hf = int((1 + self.n_fft / 2) * (cutoff_freq / self.rate))

        # frequency domain
        result["lsd"], result["lsd_hf"], result["lsd_lf"] = self.lsd(
            est_wav, target_wav, hf
        )

        for key in result:
            result[key] = float(result[key])
        return result

    def lsd(self, est, target, hf):
        """
        Function to calculate the log-spectral distortion (also for high and low frequency components)

        Args:
            est (torch.Tensor): estimated waveform
            target (torch.Tensor): target waveform
            hf (int): highcut frequency
        """
        sp = torch.log10(self.stft(est).square().clamp(1e-8))
        st = torch.log10(self.stft(target).square().clamp(1e-8))
        return (
            (sp - st).square().mean(dim=1).sqrt().mean(),
            (sp[:, hf:, :] - st[:, hf:, :]).square().mean(dim=1).sqrt().mean(),
            (sp[:, :hf, :] - st[:, :hf, :]).square().mean(dim=1).sqrt().mean(),
        )

    def sispec(self, est, target):
        # in log scale
        output, target = energy_unify(est, target)
        noise = output - target
        sp_loss = 10 * torch.log10(
            (pow_p_norm(target) / (pow_p_norm(noise) + EPS) + EPS)
        )
        return torch.sum(sp_loss) / sp_loss.size()[0]

    def ssim(self, est, target):
        if "cuda" in str(target.device):
            target, output = target.detach().cpu().numpy(), est.detach().cpu().numpy()
        else:
            target, output = target.numpy(), est.numpy()
        res = np.zeros([output.shape[0], output.shape[1]])
        for bs in range(output.shape[0]):
            for c in range(output.shape[1]):
                res[bs, c] = ssim(output[bs, c, ...], target[bs, c, ...], win_size=7)
        return torch.tensor(res)[..., None, None]


class STFTMag(nn.Module):
    def __init__(self, nfft=1024, hop=256):
        super().__init__()
        self.nfft = nfft
        self.hop = hop
        self.register_buffer("window", torch.hann_window(nfft), False)

    # x: [B,T] or [T]
    @torch.no_grad()
    def forward(self, x):
        T = x.shape[-1]
        self.window = self.window.to(x.device)
        stft = torch.stft(
            x, self.nfft, self.hop, window=self.window, return_complex=True
        )  # [B, F, TT]
        #   return_complex=False)  #[B, F, TT,2]
        mag = torch.sqrt(stft.real.pow(2) + stft.imag.pow(2))
        return mag


if __name__ == "__main__":
    import numpy as np

    au = AudioMetrics(rate=44100)
    # path1 = "old/out.wav"
    path1 = "p225_001.wav"
    # path2 = "old/target.wav"
    path2 = "p225_001.wav"
    result = au.evaluation(path2, path1, path1, 8000, device="cpu")
    print(result)

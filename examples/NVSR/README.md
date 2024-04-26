# NVSR (2kHz~44.1kHz to 44.1kHz Super-resolution)

To run our pretrained NVSR. First please install the following requirements.

```shell
pip3 install -r requirements.txt
```

Then in this folder, simply run the following command:

```shell
python3 main.py
```

## Troubleshooting
The ssr_eval is a bit out of date, some of the functions may not work properly. Here are what I have done to make it work:
1. TypeError: window_sumsquare() takes 0 positional arguments but 2 positional arguments (and 3 keyword-only arguments) were given
    ```python
    # $PATH_TO_MY_ENV$/.conda/envs/nvsr/lib/python3.10/site-packages/torchlibrosa/stft.py", line 380
    def get_ifft_window(self, n_frames, device):
            """Get tensor to be divided by signal after overlap-add."""
            ifft_window_sum = librosa.filters.window_sumsquare(
                window=self.window,         # <- add window=
                n_frames=n_frames,          # <- add n_frames=
                win_length=self.win_length,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
            )

            ifft_window_sum = np.clip(ifft_window_sum, 1e-8, np.inf)
            ifft_window_sum = torch.Tensor(ifft_window_sum).to(device)
            return ifft_window_sum
    ```
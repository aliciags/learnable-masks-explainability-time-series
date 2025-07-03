# import math
import pywt
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import upfirdn, freqz, convolve
from src.utils.sampling import upsampling_wavedec



class WaveletFilterbank():
    """
    Class to create a wavelet filterbank based on DWT for the given time series.
    """

    def __init__(self, fs:int, wavelet: str = 'haar', w_len: int = 1, level: int = None):
        # check inputs
        assert level >= 0, "Level must be greater than or equal to 0"
        assert wavelet in ['haar', 'sym',  'db', 'coif'], "Wavelet must be one of haar, db, or coif"

        # Assert w_len based on wavelet type
        if wavelet == 'db':
            assert 1 <= w_len <= 38, "Wavelet length must be between 1 and 38 for 'db' wavelet"
        elif wavelet == 'coif':
            assert 1 <= w_len <= 17, "Wavelet length must be between 1 and 24 for 'coif' wavelet"
        elif wavelet == 'sym':
            assert 2 <= w_len <= 20, "Wavelet length must be between 2 and 20 for 'sym' wavelet"

        # set attributes
        if wavelet == 'haar':
            self.wavelet = wavelet
        else:
            self.wavelet = wavelet + str(w_len)
        self.level = level

        self.coeffs = None
        self.fs = fs
        self.data = None
        self.time = None
        self.nbanks = 0
        self.banks = []

        # self._create_filterbank()

    # def _upsample_filter(self, filt, level):
    #     """
    #     Upsample filter by inserting 2^(level - 1) - 1 zeros between taps.
    #     """
    #     up_factor = 2 ** (level - 1)
    #     return upfirdn([1], filt, up=up_factor)
    
    # def _create_filterbank(self):
    #     """
    #     Create dyadically upsampled filters (hi-pass at each level + final low-pass).
    #     NOT FULLY DWT TREE since not downsampling (future problem)
    #     """
    #     wavelet = pywt.Wavelet(self.wavelet)
    #     dec_lo = np.array(wavelet.dec_lo)  # Low-pass
    #     dec_hi = np.array(wavelet.dec_hi)  # High-pass

    #     for lvl in range(1, self.level + 1):
    #         h_hi = self._upsample_filter(dec_hi, lvl)
    #         self.banks.append({'level': lvl, 'type': 'highpass', 'filter': h_hi})

    #     # Final lowpass filter at the deepest level
    #     h_lo = self._upsample_filter(dec_lo, self.level)
    #     self.banks.append({'level': self.level, 'type': 'lowpass', 'filter': h_lo})

    #     self.nbanks = len(self.banks)

    # def apply_filterbank(self, x):
    #     ## Maybe is better to use conv1D with tensors but rn it seems to complex
    #     """
    #     Apply the wavelet FIR filterbank to signal x.
    #     Returns y with shape: (len(x), n_filters)
    #     """
    #     x = x.cpu().numpy() if isinstance(x, torch.Tensor) else x
    #     y = np.zeros((*x.shape, self.nbanks))

    #     # print(f"Shape of x: {x.shape}")
    #     # print(self.nbanks)
    #     # print(f"Shape of y: {y.shape}")

    #     # for i, bank in enumerate(self.banks):
    #     #     h = bank['filter']
    #     #     filtered = convolve(x, h[None, :], mode='full') # to broadcast over channels
    #     #     # Adjust to match input length if needed
    #     #     if len(filtered) != len(x):
    #     #         center = (len(filtered) - len(x)) // 2
    #     #         filtered = filtered[center:center+len(x)]
    #     #     y[:, i] = filtered

    #     for i, bank in enumerate(self.banks):
    #         h = bank['filter']
    #         for channel_id, channel in enumerate(x):
    #             filtered = np.convolve(channel, h, mode='full')
    #             # Crop to match original length
    #             if len(filtered) != len(channel):
    #                 center = (len(filtered) - len(channel)) // 2
    #                 filtered = filtered[center:center+len(channel)]
    #             y[channel_id, :, i] = filtered

    #     return y    


    def apply_dwt_filterbank(self, data):
        """
        Apply the wavelet filterbank to the data.

        Parameters
        ----------
        data : np.ndarray or torch.Tensor
            Input data to be transformed. If a torch tensor is provided, it will be converted to numpy.
        """
        if isinstance(data, torch.Tensor):
            self.data = data.detach().cpu().numpy()
        else:
            self.data = data
        self.time = np.linspace(0, len(data) / self.fs, len(data))

        self.coeffs = pywt.wavedec(self.data, self.wavelet, level=self.level)
        self.nbanks = len(self.coeffs)

    def get_dwt_coeffs(self):
        """
        Get the DWT coefficients.
        """
        if self.coeffs is None:
            raise ValueError("DWT coefficients not computed. Call apply_dwt_filterbank() first.")
        return self.coeffs
    
    def get_wavelet_bands(self, normalize=False, rescale=False):
        upsampled_coeffs = upsampling_wavedec(len(self.data), self.coeffs)

        if rescale:
            for level, coeff in enumerate(upsampled_coeffs):
                upsampled_coeffs[level] = coeff * (2 ** level)
        if normalize:
            upsampled_coeffs = np.abs(upsampled_coeffs)
            upsampled_coeffs = (upsampled_coeffs - np.min(upsampled_coeffs)) / (np.max(upsampled_coeffs) - np.min(upsampled_coeffs) + 1e-8)

        return upsampled_coeffs
    
    def plot_filterbank(self):
        """
        Plot the magnitude frequency response of each filter in the filterbank.
        """
        plt.figure(figsize=(10, 6))

        for i, bank in enumerate(self.banks):
            h = bank['filter']
            w, H = freqz(h, worN=2048, fs=self.fs)
            label = f"Level {bank['level']} ({bank['type']})"
            plt.plot((self.fs * 0.5 / np.pi) * w, H, label=label)

        plt.title(f"Wavelet Filterbank Frequency Response ({self.wavelet})")
        plt.xlabel("Frequency [Hz]" if self.fs != 1.0 else "Normalized Frequency [×π rad/sample]")
        plt.ylabel("Gain")
        # plt.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()

    
    def plot_dwt_coeffs(self):
        """
        Plot the DWT coefficients.
        """
        fig, axes = plt.subplots(self.level+1, 1, figsize=(10, 10))

        axes[0].plot(self.data, label="Original Signal")
        axes[0].set_title("Original Signal")

        for i, coeff in enumerate(self.coeffs):
            # if i == 0:
            #     axes[i + 1].plot(coeff, label="Approximation Coefficients CA")
            #     axes[i + 1].set_title("Approximation Coefficients CA")
            # else:
            j = self.level # - i
            axes[i + 1].plot(coeff, label=f"Detail Coefficients CD{j}")
            axes[i + 1].set_title(f"Detail Coefficients CD{j}")

            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')

        plt.tight_layout()

        # save the figure
        plt.savefig(f'/public/wavelet/dwt_singal_filtered_{self.wavelet}_{self.level}.png')

    def get_filter_response(self, mask=None, rescale=True, normalize=True):
        """
        Compute the filtered signal using masked wavelet coefficients.
        Supports time-varying mask across DWT levels.

        Parameters
        ----------
        mask : np.ndarray
            Mask with shape (signal_length, n_filters) or (1, signal_length, n_filters)
        rescale : bool
            Rescale the coefficients to the original signal length.
        normalize : bool
            Normalize the coefficients to the range [0, 1].

        Returns
        -------
        np.ndarray
            Filtered response with shape (1, signal_length)
        """
        if self.coeffs is None:
            raise ValueError("Wavelet coefficients not computed.")

        # Convert to array and upsample to original signal length
        upsampled_coeffs = self.get_wavelet_bands(rescale=rescale, normalize=normalize)
        upsampled_coeffs = np.array(upsampled_coeffs)

        # Move to shape (1, signal_length, n_filters) for broadcasting across channels
        upsampled_coeffs = np.moveaxis(upsampled_coeffs, 0, -1)[np.newaxis, :, :] 

        # Handle mask shape
        if mask.ndim == 2:
            mask = mask[np.newaxis, :, :]  # add channel dim
        
        # element-wise masking
        response = upsampled_coeffs * mask if mask is not None else upsampled_coeffs
        
        return response # .sum(axis=-1)
    
    def get_collect_filter_response(self, mask=None):
        """
        Compute the collective frequency response of the wavelet filterbank,
        optionally modulated by a learned mask.

        Parameters
        ----------
        mask : np.ndarray or None
            A mask with shape (n_samples, n_filters) or (n_filters,). If None, all filters are unmasked.

        Returns
        -------
        np.ndarray
            Combined frequency response per sample. Shape: (n_freq_bins, n_samples)
        """
        worN = len(np.fft.rfftfreq(self.fs, 1/self.fs))  # Frequency bins for FFT
        mask = np.array(mask) if mask is not None else np.ones((1, self.nbanks))
        
        if mask.ndim == 1:
            mask = mask[np.newaxis, :]  # Make shape (1, n_filters)

        n_samples = mask.shape[0]
        collect_freq_resp = np.zeros((worN, n_samples, self.nbanks))

        for i, bank in enumerate(self.banks):
            h = bank['filter']
            w, H = freqz(h, worN=worN, fs=self.fs)
            H_mag = np.abs(H)
            for j in range(n_samples):
                collect_freq_resp[:, j, i] = H_mag * mask[j, i]

        # not summing over filters to keep granularity for each band
        return collect_freq_resp

    def plot_dwt_scaleogram_freq(self):
        """
        Plot the scaleogram of the Discrete Wavelet Transform (DWT).
        """

        # Compute frequency bands
        freq_bands = [self.fs /  (2 ** (j + 1)) for j in range(self.level)]   # bands
        freq_bands.append(0)                                        # lowest frequency
        print(freq_bands)

        # flip the frequency bands
        freq_bands = freq_bands[::-1]

        # Prepare the scaleogram
        # scaleogram = []
        # for i, coeff in enumerate(self.coeffs):
        #     factor = math.ceil(len(self.data)/len(coeff))
        #     # Upsample coefficients to match the original signal length
        #     upsampled = np.repeat(coeff, factor)[:len(self.data)]
        #     scaleogram.append(upsampled)
        scaleogram = np.array(upsampling_wavedec(self.coeffs))
        scaleogram = scaleogram[:, :(len(self.data))-1]

        # Plot the scaleogram
        plt.figure(figsize=(10, 6))
        plt.pcolormesh(self.time, freq_bands, np.abs(scaleogram), cmap='viridis')
        plt.colorbar(label='Amplitude')
        plt.title('DWT Scaleogram')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')

        # save the figure
        plt.savefig(f'/public/wavelet/dwt_scaleogram_{self.wavelet}.png')


    def plot_dwt_scaleogram_lvl(self):
        """
        Plot the scaleogram of the Discrete Wavelet Transform (DWT).
        """

        # Prepare the scaleogram
        # scaleogram = []
        # for i, coeff in enumerate(self.coeffs):
        #     factor = math.ceil(len(self.data)/len(coeff))
        #     # Upsample coefficients to match the original signal length
        #     upsampled = np.repeat(coeff, factor)[:len(self.data)]
        #     scaleogram.append(upsampled)
        scaleogram = np.array(upsampling_wavedec(self.coeffs))
        scaleogram = scaleogram[:, :(len(self.data))-1]

        # Calculate the frequencies for each level (scales are powers of 2, so freqs are powers of 2)
        freqs_dwt = np.logspace(start=0, stop=self.level+1, num=self.level+2, base=2)

        # Plot the scaleogram
        plt.figure(figsize=(10, 6))
        plt.pcolormesh(self.time, freqs_dwt, np.abs(scaleogram), cmap='viridis')
        plt.colorbar(label='Amplitude')
        plt.title('DWT Scaleogram')
        plt.yscale('log', base=2)
        plt.yticks(ticks=freqs_dwt, labels=[f'{int(i)}' for i in range(len(freqs_dwt))])
        plt.xlabel('Time (s)')
        plt.ylabel('Decomposition Level')

        # save the figure
        plt.savefig(f'/public/wavelet/dwt_scaleogram_{self.wavelet}_{self.level}.png')

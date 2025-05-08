import math
import matplotlib.pyplot as plt
import numpy as np
import pywt
import torch


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
        self.nfilters = 0


    def apply_dwt_filterbank(self, data):
        """
        Apply the wavelet filterbank to the image.
        """
        if isinstance(data, torch.Tensor):
            self.data = data.detach().cpu().numpy()
        else:
            self.data = data
        self.time = np.linspace(0, len(data) / self.fs, len(data))

        self.coeffs = pywt.wavedec(self.data, self.wavelet, level=self.level)
        self.nfilters = len(self.coeffs)

    def get_dwt_coeffs(self):
        """
        Get the DWT coefficients.
        """
        # if self.coeffs is None:
        #     raise ValueError("DWT coefficients not computed. Call apply_dwt_filterbank() first.")
        return self.coeffs
    
    def get_wavelet_bands(self, normalize=True, rescale=True):
        upsampled_coeffs = []

        for level, coeff in enumerate(self.coeffs):
            factor = int(np.ceil(len(self.data) / len(coeff)))
            upsampled = np.repeat(coeff, factor)[:len(self.data)]
            
            if rescale:
                upsampled *= 2 ** level
            if normalize:
                upsampled = (upsampled - np.min(upsampled)) / (np.max(upsampled) - np.min(upsampled) + 1e-8)

            upsampled_coeffs.append(upsampled)

        return np.array(upsampled_coeffs)
    
    def plot_dwt_coeffs(self):
        """
        Plot the DWT coefficients.
        """
        fig, axes = plt.subplots(self.level+2, 1, figsize=(10, 10))

        axes[0].plot(self.data, label="Original Signal")
        axes[0].set_title("Original Signal")

        for i, coeff in enumerate(self.coeffs):
            if i == 0:
                axes[i + 1].plot(coeff, label="Approximation Coefficients CA")
                axes[i + 1].set_title("Approximation Coefficients CA")
            else:
                j = self.level - i
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

    def plot_dwt_scaleogram_freq(self):
        """
        Plot the scaleogram of the Discrete Wavelet Transform (DWT).
        """

        # Compute frequency bands
        freq_bands = [self.fs /  (2 ** (j + 1)) for j in range(self.level)]   # bands
        freq_bands.append(self.fs / (2 ** self.level))                        # Nyquist
        freq_bands.append(0)                                        # lowest frequency
        print(freq_bands)

        # flip the frequency bands
        freq_bands = freq_bands[::-1]

        # Prepare the scaleogram
        scaleogram = []
        for i, coeff in enumerate(self.coeffs):
            factor = math.ceil(len(self.data)/len(coeff))
            # Upsample coefficients to match the original signal length
            upsampled = np.repeat(coeff, factor)[:len(self.data)]
            scaleogram.append(upsampled)
        scaleogram = np.array(scaleogram)
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
        scaleogram = []
        for i, coeff in enumerate(self.coeffs):
            factor = math.ceil(len(self.data)/len(coeff))
            # Upsample coefficients to match the original signal length
            upsampled = np.repeat(coeff, factor)[:len(self.data)]
            scaleogram.append(upsampled)
        scaleogram = np.array(scaleogram)
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

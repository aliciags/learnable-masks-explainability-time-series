import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

class Filterbank:
    def __init__(self, n_taps: int,  n_filters: int, sample_freq: int, time_len: int, bandwidth: float = None):
        # check inputs
        assert n_taps > 0, "Number of taps must be greater than 0"
        assert n_taps % 2 == 1, "Number of taps must be odd for best symmetry"
        assert n_filters >= 2, "Number of filters must be greater or equal to 2"
        assert sample_freq > 0, "Sample frequency must be greater than 0"

        # set attributes
        self.n_taps = n_taps
        self.n_filters = n_filters
        self.sample_freq = sample_freq
        self.group_delay = (n_taps - 1) // 2  # filter is symmetric
        self.time_len = time_len
        self.bank = []

        if bandwidth is None:
            self.bandwidth = (sample_freq / 2) / n_filters
        else:
            self.bandwidth = bandwidth

        # create filterbank
        self.create_filterbank()

    def create_filterbank(self):
        """
        Create filterbank with n_filters filters.
        """
        # first filter pass low
        h = signal.firwin(self.n_taps, self.bandwidth, fs=self.sample_freq, pass_zero="lowpass")
        self.bank.append(h)
        band_start = self.bandwidth

        # middle filters pass band
        for i in range(1, self.n_filters - 1):
            h = signal.firwin(self.n_taps, [band_start, band_start + self.bandwidth], fs=self.sample_freq, pass_zero="bandpass")
            self.bank.append(h)
            band_start += self.bandwidth

        # last filter pass high
        h = signal.firwin(self.n_taps, band_start, fs=self.sample_freq, pass_zero="highpass")
        self.bank.append(h)

    def get_filterbank(self):
        """
        Get the filterbank.
        """
        return self.bank
    
    def plot_filterbank(self):
        """
        Plot the filterbank.
        """
        # plot filterbank in a single plot
        plt.figure()
        cmap = get_cmap('hsv', len(self.bank))
        for i, h in enumerate(self.bank):
            w, h_response = signal.freqz(h, worN=2000)
            plt.plot((self.sample_freq * 0.5 / np.pi) * w, np.abs(h_response), color=cmap(i))
        plt.title("Filterbank")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Gain")
        plt.show()

    def apply_filterbank(self, data, time_axis: int =-1, adjust_delay: bool = True):
        """
        Apply filterbank to data.
        
        Parameters
        ----------
        data : np.ndarray
            Input data with shape (N, C, T) where N is number of sequences, C the number of channels, T is number of time points.
        time_axis : int
            The time axis of the data.

        Returns
        -------
        np.ndarray
            The output of the filterbank.
        """
        # apply filterbank to data
        y = np.zeros((*data.shape, self.n_filters))
        for i, h in enumerate(self.bank):
            y_tmp = signal.lfilter(h, 1, data, axis=time_axis)
            if adjust_delay:
                y_tmp = np.roll(y_tmp, -self.group_delay, axis=time_axis)
            y[..., i] = y_tmp
        return y 
    
    def get_filter_response(self, mask = None):
        """
        Get the filter response based on the mask on the frequency domain.

        Parameters
        ----------
        mask : np.ndarray
            Mask with shape (N, T) or (T, ) to apply to the data.

        Returns
        -------
        np.ndarray
            The filter response.
        """
        worN = len(np.fft.rfftfreq(self.time_len, 1 / self.sample_freq))
        if len(mask.shape) == 1:
            mask = mask[np.newaxis, :]
        response = np.zeros((worN, *mask.shape))
        for i in range(mask.shape[0]):
            for j, h in enumerate(self.bank):
                w, h_response = signal.freqz(h, 1, worN=worN)
                h_response = np.abs(h_response) * mask[i, j] if mask is not None else np.abs(h_response)
                response[:, i, j] = h_response

        return np.sum(response, axis=-1)
    
    def plot_filter_response(self, ax = None,  mask = None, time_axis: int =-1):
        # I believe the wordN should be based on the mask if it passed so the multiplication can be computed
        """
        Plot the filter response based on the mask.

        Parameters
        ----------
        mask : np.ndarray
            Mask with shape (N, T) or (T, ) to apply to the data.
        time_axis : int
            The time axis of the data.
        """
        # set axis
        if ax is None:
            fig, ax = plt.subplots()

        # get the filter response
        response = np.zeros((2000, self.n_filters))
        for i, h in enumerate(self.bank):
            w, h_response = signal.freqz(h, 1, worN=2000)
            h_response = np.abs(h_response) * mask if mask is not None else np.abs(h_response)
            response[:, i] = h_response
        
        # plot filter response
        ax.plot((self.sample_freq * 0.5 / np.pi) * w, np.sum(response, axis=1))
        ax.set_title("Filter Response")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Gain")
        plt.show()
    
    def forward(self, data, mask = None, time_axis: int =-1, adjust_delay: bool = True):
        """
        Apply filterbank to data and apply mask if provided.

        Parameters
        ----------
        data : np.ndarray
            Not 100% sure about the shape of the data.
            Input data with shape (N, C, T) where N is number of samples, C is number of channels, T is number of time points.
        mask : np.ndarray
            Mask with shape (N, T) or (T, ) to apply to the data.
        time_axis : int
            The time axis of the data.

        Returns
        -------
        np.ndarray
            The output of the filterbank with mask applied summed over the filter dimension.
        """
        # apply filterbank to data
        y = self.apply_filterbank(data, time_axis, adjust_delay)

        # apply mask if provided
        if mask is not None:
            # if the mask dimension is 1
            if len(mask.shape) == 1:
                mask = mask[np.newaxis, :]  # convert (N, ) to (1, N)
                y = y * mask
            # if the mask dimension is 2
            else:
                y = y * mask[:, np.newaxis, :]

        return np.sum(y, axis=-1)
    
    

import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

class Filterbank:
    def __init__(self, n_taps: int,  n_filters: int, sample_freq, bandwidth: float = None):
        # check inputs
        assert n_taps > 0, "Number of taps must be greater than 0"
        assert n_taps % 2 == 1, "Number of taps must be odd for best symmetry"
        assert n_filters >= 2, "Number of filters must be greater or equal to 2"
        assert sample_freq > 0, "Sample frequency must be greater than 0"

        # set attributes
        self.n_taps = n_taps
        self.n_filters = n_filters
        self.sample_freq = sample_freq
        self.bank = []

        if bandwidth is None:
            self.bandwidth = (sample_freq / 2) / n_filters
        else:
            self.bandwidth = bandwidth

    def create_filterbank(self):
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
        return self.bank
    
    def plot_filterbank(self):
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
        

import pywt
import numpy as np
import matplotlib.pyplot as plt

def plot_wavelet_filters(wavelet_name, figsize=(12, 8), plot_frequency_response=False):
    """
    Plot the wavelet filters (low-pass and high-pass) for a given wavelet.
    
    Parameters:
    -----------
    wavelet_name : str
        Name of the wavelet to plot (e.g., 'db4', 'haar', 'sym4')
    figsize : tuple, optional
        Figure size (width, height) in inches
    plot_frequency_response : bool, optional
        Whether to plot the frequency response of the filters
    """
    # Get the wavelet object
    wavelet = pywt.Wavelet(wavelet_name)
    
    # Create figure and subplots
    if plot_frequency_response:
        fig, axes = plt.subplots(2, 2, figsize=figsize)
    else:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Get the filters
    low_pass = wavelet.dec_lo
    high_pass = wavelet.dec_hi
    
    # Plot low-pass filter
    if plot_frequency_response:
        axes[0, 0].plot(low_pass)
        axes[0, 0].set_title('Low-pass Filter (Scaling Function)')
        axes[0, 0].grid(True)
        axes[0, 0].set_xlabel('Sample Index')
        axes[0, 0].set_ylabel('Amplitude')
    else:
        axes[0].plot(low_pass)
        axes[0].set_title('Low-pass Filter (Scaling Function)')
        axes[0].grid(True)
        axes[0].set_xlabel('Sample Index')
        axes[0].set_ylabel('Amplitude')
    
    # Plot high-pass filter
    if plot_frequency_response:
        axes[0, 1].plot(high_pass)
        axes[0, 1].set_title('High-pass Filter (Wavelet Function)')
        axes[0, 1].grid(True)
        axes[0, 1].set_xlabel('Sample Index')
        axes[0, 1].set_ylabel('Amplitude')
    else:
        axes[1].plot(high_pass)
        axes[1].set_title('High-pass Filter (Wavelet Function)')
        axes[1].grid(True)
        axes[1].set_xlabel('Sample Index')
        axes[1].set_ylabel('Amplitude')
    
    if plot_frequency_response:
        # Compute frequency response
        N = len(low_pass)
        freq = np.fft.fftfreq(N)
        
        # Plot frequency response of low-pass filter
        axes[1, 0].plot(freq, np.abs(np.fft.fft(low_pass)) / N)
        axes[1, 0].set_title('Frequency Response - Low-pass')
        axes[1, 0].grid(True)
        axes[1, 0].set_xlabel('Normalized Frequency')
        axes[1, 0].set_ylabel('Magnitude')
        axes[1, 0].set_xlim(-0.5, 0.5)
        
        # Plot frequency response of high-pass filter
        axes[1, 1].plot(freq, np.abs(np.fft.fft(high_pass)) / N)
        axes[1, 1].set_title('Frequency Response - High-pass')
        axes[1, 1].grid(True)
        axes[1, 1].set_xlabel('Normalized Frequency')
        axes[1, 1].set_ylabel('Magnitude')
        axes[1, 1].set_xlim(-0.5, 0.5)
    
    # Add main title
    plt.suptitle(f'Wavelet Filters for {wavelet_name} Wavelet', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    return fig, axes


def plot_scaleogram_wav_heatmap(signal, attribution, label:int, fs=100, wavelet='db1'):
    # print(attribution.shape)

    # Compute frequency bands
    freq_bands = [fs /  (2 ** (j + 1)) for j in range(len(attribution[0]))]   # bands
    freq_bands.append(0)                                        # lowest frequency
    freq_bands = freq_bands[::-1]                               # reverse the order
    # print(freq_bands)

    time = np.linspace(0, signal / fs, signal)

    # normalize attribution scores to [0,1]
    attribution = np.abs(attribution)
    attr_norm = (attribution - attribution.min()) / (attribution.max() - attribution.min() + 1e-10)

    # Create a 2D grid for heatmap
    time_grid, freq_grid = np.meshgrid(time, freq_bands)
    # make attr x one dim smaller
    attr_norm = attr_norm[1:,:]

    # Plot the scaleogram
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(time_grid, freq_grid, attr_norm.T, shading='auto', cmap='Greens')
    plt.colorbar(label="Normalized Attribution Score")
    plt.title(f'Attributions of class {label} learned through wavelet {wavelet}')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')

    plt.savefig(f'/Users/alicia/Documents/Master/TFM/master_thesis/graphics/attr_5_{wavelet}_{label}.png')

    plt.show()

# plot scaleograms

def plot_dwt_scaleogram(coeffs, w='db1', fs=16, label=0):
    """
    coeffs: list of coefficients
    fs: sampling frequency
    """
    # Create a figure
    plt.figure(figsize=(10, 6))

    # compute the frequency bands
    freq_bands = [fs / (2 ** (j + 1)) for j in range(len(coeffs[0]))]   # bands
    freq_bands.append(0)                                             # lowest frequency
    freq_bands = freq_bands[::-1]                                    # reverse the order
    
    scaleogram = np.array(coeffs)
    # scaleogram = scaleogram[:, :]
    scaleogram = np.abs(scaleogram)
    
    # normalize the scaleogram
    scaleogram = (scaleogram - np.min(scaleogram)) / (np.max(scaleogram) - np.min(scaleogram) + 1e-10)

    # Create a meshgrid for time and frequency
    time = np.linspace(0, 1, fs)
    time = np.concatenate([time, [time[-1] + (1/fs)]])

    plt.figure(figsize=(10, 6))
    plt.pcolormesh(time, freq_bands, scaleogram.T, shading='auto', cmap='viridis')
    plt.colorbar(label='Magnitude')
    plt.title(f'Scaleogram of class {label} of wavelet {w}')
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')

    plt.tight_layout()
    plt.show()
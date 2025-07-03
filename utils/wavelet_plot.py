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

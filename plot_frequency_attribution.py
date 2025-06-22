import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List


def plot_frequency_attribution(
    masks: List[np.ndarray],
    labels: List[int],
    fs: int = 100,
    data_length: int = 3000,
    max_level: int = 5,
    figsize: Tuple[int, int] = (20, 5)
):
    """
    Plot the frequency band attribution for each sleep stage.
    
    Args:
        masks: List of averaged wavelet masks for each sleep stage
        labels: List of corresponding sleep stage labels
        fs: Sampling frequency
        data_length: Length of the data in samples
        max_level: Maximum wavelet decomposition level
        figsize: Figure size (width, height)
    """
    # Calculate frequency bands based on wavelet decomposition
    # For wavelet transform, each level represents a different frequency band
    # The frequency bands are approximately: [f/2^(level+1), f/2^level]
    freq_bands = [fs /  (2 ** (j + 1)) for j in range(max_level)]   # bands
    freq_bands.append(0)                                            # lowest frequency
    freq_bands = freq_bands[::-1]    
    
    # Create subplots in a single row
    fig, axes = plt.subplots(1, len(masks), figsize=figsize, sharey=True)
    
    # Find the maximum absolute value for normalization
    max_abs_val = max(np.max(np.abs(np.sum(mask, axis=0))) for mask in masks)
    
    for i, (mask, label) in enumerate(zip(masks, labels)):
        # Integrate over time dimension (axis=1)
        freq_attribution = np.sum(mask, axis=0)
        
        # Normalize the attribution scores
        freq_attribution = freq_attribution / max_abs_val
        
        # Plot the frequency attribution
        ax = axes[i]
        
        # Create a line plot with frequency bands
        ax.plot(freq_bands, freq_attribution, marker='o', linestyle='-', color='blue')
        
        # Format the plot
        ax.set_title(f'Sleep Stage {label}', pad=10)
        ax.set_xlabel('Frequency (Hz)')
        
        # Set consistent limits
        ax.set_xlim(0, fs/2)
        ax.set_ylim(-1.1, 1.1)  # Slightly larger than normalized range
        
        # Add horizontal grid lines
        ax.grid(True, axis='both', linestyle='--', alpha=0.7)
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Add a horizontal line at y=0 for reference
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # Add vertical lines to separate frequency bands
        for j in range(len(freq_attribution) - 1):
            ax.axvline(x=j + 0.5, color='gray', linestyle='--', alpha=0.3)
    
    # Add common x-axis label
    plt.xlabel('Frequency Band')
    plt.tight_layout()
    plt.show()
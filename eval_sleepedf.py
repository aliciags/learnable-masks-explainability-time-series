import pickle
import importlib
import numpy as np
import matplotlib.pyplot as plt

from scipy.fftpack import fft, fftfreq
from src.attribution.flextime.filterbank import Filterbank

from physioex.physioex.data import PhysioExDataModule

target_package = "physioex.train.networks.utils.target_transform"
target_class = "get_mid_label"
target = getattr(importlib.import_module(target_package), target_class)

datamodule = PhysioExDataModule(
    datasets=["sleepedf"],     # list of datasets to be used
    batch_size=128,            # batch size for the DataLoader
    preprocessing="raw",       # preprocessing method
    selected_channels=["EEG"], # channels to be selected
    sequence_length=7,         # length of the sequence
    target_transform= target,  # since seq to epoch, target seq
    num_workers = 10,          # number of parallel workers
    data_folder = "/work3/s241931/data/"    # path to the data folder
)

# get the test DataLoaders
test_loader = datamodule.test_dataloader()

print(len(test_loader)) # number of batches in the test set

# get the masks
with open("public/sleepedf/masks_sleepEDF_7_2.pkl", "rb") as f:
    masks = pickle.load(f)

# save the scores
with open("public/sleepedf/scores_sleepEDF_7_2.pkl", "rb") as f:
    scores = pickle.load(f)

# create the filterbank
filterbank = Filterbank(n_taps=501, n_filters=16, sample_freq=100, time_len=3000)



def plot_freq_heatmap(signal, attribution, i:int, fs=100):
    # Compute FFT
    N = len(signal)
    freq = fftfreq(N, d=1/fs)  # Frequency bins
    magnitude = np.abs(fft(signal))  # Magnitude of FFT

    # Keep only positive frequencies
    pos_mask = freq >= 0
    freq = freq[pos_mask]
    magnitude = magnitude[pos_mask]

    # Normalize attribution scores to [0,1]
    attr_resized = np.interp(freq, np.linspace(freq.min(), freq.max(), len(attribution)), attribution)
    attr_norm = (attr_resized - np.min(attr_resized)) / (np.max(attr_resized) - np.min(attr_resized) + 1e-10)

    # Create a 2D grid for heatmap
    freq_grid, mag_grid = np.meshgrid(freq, magnitude)
    attr_grid = np.tile(attr_norm[:-1], (len(magnitude) - 1, 1))  # Repeat attribution scores along the magnitude axis

    # Plot heatmap
    plt.figure(figsize=(10, 5))
    plt.plot(freq, magnitude, label="FFT Magnitude", color='black', alpha=0.6)
    plt.pcolormesh(freq_grid, mag_grid, attr_grid, shading='auto', cmap='Greens')
    plt.colorbar(label="Normalized Attribution Score")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude")
    plt.title("Attribution Heatmap in Frequency Domain")
    plt.savefig(f"public/plots/attr_heatmap_freq_{i}.png")
    plt.close()


# plot the scores as a heatmap on the original signal
for x, y in test_loader:
    # get the first sequence
    x = x.numpy()

    # plot 10 samples
    for i in range(1, 11):
        plot_freq_heatmap(x[i][0][0], scores[0][i], i)
    break
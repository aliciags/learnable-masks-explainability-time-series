# simulate data
import os
import numpy as np
from src.data_gen import generate_signal

np.random.seed(33)  # For reproducibility

num_samples = 5000  # Number of samples per class
fs = 1000           # Sampling rate
T = 1               # Duration in seconds
N = fs * T          # Number of samples

t = np.linspace(0, T, N)

freqs_1 = [50, 100, 320, 60, 480]     # salient frequencies
freqs_2 = [50, 480, 320, 60, 100]     # salient frequencies
times = [0.1, 0.2, 0.4, 0.4, 0.7]     # salient times
ampls = [1, 0.9, 0.8, 0.8, 1]         # amplitudes

# train data
# folder = "./data/synthetic/train/"
# test data
folder = "./data/synthetic/test_1/"

# create the folder if it does not exist
if not os.path.exists(folder):
    os.makedirs(folder)

# change range from 5 to 2 depending if train or test
for i in range(1):
# for i in range(2):
    # generate sequences of samples
    samples_0 = [generate_signal(frequencies=freqs_1, times=times, amplitudes=ampls, fs=fs, T=T, noise_level=0.1) for _ in range(num_samples)]
    samples_1 = [generate_signal(frequencies=freqs_2, times=times, amplitudes=ampls, fs=fs, T=T, noise_level=0.1) for _ in range(num_samples)]

    # create the npy files
    np.save(folder + "samples_0_%d.npy" % i, samples_0)
    np.save(folder + "samples_1_%d.npy" % i, samples_1)
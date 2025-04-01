# simulate data
import os
import numpy as np
from src.data_gen import generate_signal

num_samples = 5000  # Number of samples per class
fs = 100            # Sampling rate
T = 30              # Duration in seconds
N = fs * T          # Number of samples

t = np.linspace(0, T, N)

freqs = [5, 20]     # salient frequencies
times = [7, 21]     # salient times
ampls = [1, 0.7]    # amplitudes

folder = "./data/synthetic/test/"

# create the folder if it does not exist
if not os.path.exists(folder):
    os.makedirs(folder)


for i in range(2):
    # generate five sequences of samples
    samples_0 = [generate_signal(frequencies=freqs, times=times, amplitudes=ampls, fs=fs, T=T) for _ in range(num_samples)]
    samples_1 = [generate_signal(frequencies=freqs[::-1], times=times, amplitudes=ampls, fs=fs, T=T) for _ in range(num_samples)]

    # create the npy files
    np.save(folder + "samples_0_%d.npy" % i, samples_0)
    np.save(folder + "samples_1_%d.npy" % i, samples_1)
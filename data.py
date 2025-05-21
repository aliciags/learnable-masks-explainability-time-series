# simulate data
import os
import numpy as np
from src.data_gen import generate_signal

SEED = 42

np.random.seed(SEED)  # For reproducibility

num_samples = 5000  # Number of samples per class
fs = 100            # Sampling rate
T = 1               # Duration in seconds
N = fs * T          # Number of samples

t = np.linspace(0, T, N)

freqs_0 = [8]     # salient frequencies
freqs_1 = [30]    # salient frequencies
times_0 = [0.3]   # salient times
times_1 = [0.7]   # salient times
ampls = [1]       # amplitudes

# folders for train and test data
folder_train = "./data/synthetic/train_3/"
folder_test = "./data/synthetic/test_3/"

# create the folder if it does not exist
if not os.path.exists(folder_train):
    os.makedirs(folder_train)

if not os.path.exists(folder_test):
    os.makedirs(folder_test)

# genrate the data
for i in range(2):
    # generate sequences of samples
    samples_0 = [generate_signal(frequencies=freqs_0, times=times_0, amplitudes=ampls, fs=fs, T=T, noise_level=0.001, jitter=0.0) for _ in range(num_samples)]
    samples_1 = [generate_signal(frequencies=freqs_1, times=times_1, amplitudes=ampls, fs=fs, T=T, noise_level=0.001, jitter=0.0) for _ in range(num_samples)]

    # create the npy files
    np.save(folder_train + "samples_0_%d.npy" % i, samples_0)
    np.save(folder_train + "samples_1_%d.npy" % i, samples_1)

for i in range(1):
    # generate sequences of samples
    samples_0 = [generate_signal(frequencies=freqs_0, times=times_0, amplitudes=ampls, fs=fs, T=T, noise_level=0.001, jitter=0.0) for _ in range(num_samples)]
    samples_1 = [generate_signal(frequencies=freqs_1, times=times_1, amplitudes=ampls, fs=fs, T=T, noise_level=0.001, jitter=0.0) for _ in range(num_samples)]

    # create the npy files
    np.save(folder_test + "samples_0_%d.npy" % i, samples_0)
    np.save(folder_test + "samples_1_%d.npy" % i, samples_1)
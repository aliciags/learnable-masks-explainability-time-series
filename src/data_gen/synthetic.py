import numpy as np

# Function to generate varied signals for classification
def generate_signal(frequencies, times, amplitudes, noise_level=0.1, jitter=0.2, fs: int = 100, T: int = 30):
    """
    Generate a signal with multiple frequency components at different times and amplitudes.
    
    Parameters
    ----------
    frequencies : list
        List of frequencies of the components.
    times : list
        List of times where the components are located.
    amplitudes : list
        List of amplitudes of the components.
    noise_level : float
        Standard deviation of the noise.
    jitter : float
        Maximum relative variation of the frequency and amplitude.
    fs : int
        Sampling frequency.
    T : int
        Duration of the signal.
    
    Returns
    -------
    signal : array
        The generated signal.
    """

    N = fs * T  # Number of samples
    t = np.linspace(0, T, N)

    signal = np.zeros_like(t)
    
    for f, t_center, A in zip(frequencies, times, amplitudes):
        # Introduce random variations
        f_var = f * np.random.uniform(1 - jitter, 1 + jitter)  # Frequency variation
        t_var = t_center + np.random.uniform(-0.1, 0.1)  # Timing jitter
        A_var = A * np.random.uniform(0.8, 1.2)  # Amplitude variation
        phase = np.random.uniform(0, 2*np.pi)  # Random phase
        
        # Gaussian window to localize the frequency component
        window = np.exp(-((t - t_var)**2) / (2 * (T/10)**2))  
        signal += A_var * np.sin(2 * np.pi * f_var * t + phase) * window
    
    # Add different noise to each sample
    signal += noise_level * np.random.randn(N)  
    
    return signal
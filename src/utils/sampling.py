import re
import math
import numpy as np

def upsampling_wavedec(length, coeffs):
    """
    Upsample the coefficients of the multilevel wavelet transform 
    to match the original signal length.

    Parameters
    ----------
    length : int
        Length of the original signal.
    coeffs : list
        Coefficients of the wavelet transform.

    Returns
    -------
    np.ndarray
        Upsampled coefficients.
    """
    upsampled_coeffs = []
    for coeff in coeffs:
        factor = math.ceil(length/len(coeff))

        # Upsample coefficients to match the original signal length
        upsampled = np.repeat(coeff, factor)[:length]
        
        upsampled_coeffs.append(upsampled)
    return np.array(upsampled_coeffs)

def downsample_wavedec(length, coeffs, wavelet, filter_length):
    """
    Downsample the coefficients of the wavelet transform to match the original signal length.

    Parameters
    ----------
    length : int
        Length of the original signal.
    coeffs : list
        Coefficients of the wavelet transform.
    wavelet : str
        Type of wavelet used.
    filter_length : int
        Length of the filter used in the wavelet transform.
    
    Returns
    -------
    list
        Downsampled coefficients.
    """
    coeffs = coeffs[::-1]

    if 'db' in wavelet or 'sym' in wavelet:
        filter_length = 2 * filter_length
    elif 'coif' in wavelet:
        filter_length = 6 * filter_length

    downsampled_coeffs = []
    level_length = length
    for i, coeff in enumerate(coeffs):
        if i != len(coeffs) - 1:
            level_length = math.floor((level_length + filter_length - 1) / 2)
        indices = np.ceil(np.linspace(0, len(coeff) - 1, level_length)).astype(int)
        c = coeff[indices]

        downsampled_coeffs.append(c)
    return downsampled_coeffs[::-1]

def split_string(s):
    # Use regex to find all sequences of digits and non-digits
    parts = re.findall(r'\d+|[^\d\s]+', s)
    return parts
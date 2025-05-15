import math
import numpy as np

def upsampling_wavedec(coeffs):
    """
    Upsample the coefficients of the multilevel wavelet transform 
    to match the original signal length.
    """
    length = len(coeffs[-1]) * 2

    upsampled_coeffs = []
    for coeff in coeffs:
        factor = math.ceil(length/len(coeff))

        # Upsample coefficients to match the original signal length
        upsampled = np.repeat(coeff, factor)[:length]
        
        upsampled_coeffs.append(upsampled)
    return np.array(upsampled_coeffs)

def downsample_wavedec(coeffs):
    """
    Downsample the coefficients of the wavelet transform to match the original signal length.
    """
    coeffs = coeffs[::-1]

    downsampled_coeffs = []
    for i, coeff in enumerate(coeffs):
        c = coeff[1::(2**(i+1))]

        downsampled_coeffs.append(np.asarray(c))
    return downsampled_coeffs[::-1]
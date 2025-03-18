from .flextime import *
from .wavelet import *

def compute_attribution(method: str, model, test_loader, device, args):
    if method == 'wavelet':
        pass
    elif method == 'flextime':
        pass
    else:
        raise ValueError('Unknown method: {}. Please choose from "wavelet", "flextime"'.format(method))
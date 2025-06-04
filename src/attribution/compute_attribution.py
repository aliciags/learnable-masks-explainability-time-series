from .flextime import compute_flextime_attribution
from .wavelet import compute_wavelet_attribution

def compute_attribution(method: str, model, test_loader, args, device: str = 'mps', verbose: bool = True, normalize: bool = False, regularization: str = 'l1'):
    if method == 'wavelet':
        masks, scores = compute_wavelet_attribution(model, test_loader, filterbank_params = {'wavelet': args.wavelet, 'w_len': args.len_w, 'level': args.level, 'fs': args.sample_freq}, device = device, verbose=verbose, normalize=normalize, regularization=regularization)
    elif method == 'flextime':
        masks, scores = compute_flextime_attribution(model, test_loader, filterbank_params = {'n_taps': args.n_taps, 'n_filters': args.n_filters, 'sample_freq': args.sample_freq, 'time_len': args.time_len}, device = device, verbose=verbose)
    else:
        raise ValueError('Unknown method: {}. Please choose from "wavelet", "flextime"'.format(method))
    
    return masks, scores
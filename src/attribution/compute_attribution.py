from .flextime import compute_flextime_attribution

def compute_attribution(method: str, model, test_loader, args, device: str = 'mps'):
    if method == 'wavelet':
        pass
    elif method == 'flextime':
        # eventually pass filterbank_params
        masks, scores = compute_flextime_attribution(model, test_loader, filterbank_params = {'n_taps': args.n_taps, 'n_filters': args.n_filters, 'sample_freq': args.sample_freq, 'time_len': args.time_len}, device = device)
    else:
        raise ValueError('Unknown method: {}. Please choose from "wavelet", "flextime"'.format(method))
    
    return masks, scores
import torch
import numpy as np
from src.attribution.wavelet import WaveletFilterbank, WaveletMask
from src.utils.sampling import upsampling_wavedec


def compute_wavelet_attribution( model, 
                                 dataloader, 
                                 filterbank_params = {'wavelet': 'db', 'w_len': 1, 'fs': 16, 'level': 4}, 
                                 device:str = 'cpu', 
                                 verbose:bool = True,
                                 regularization:str = 'l1'):
    # define mask
    masks = []
    scores = []
    losses = []

    # create filterbank
    filterbank = WaveletFilterbank(**filterbank_params)

    # create FLEXtime mask
    mask_opt = WaveletMask(model, filterbank, device=device, regularization=regularization) 

    for i, batch in enumerate(dataloader):
        batch_scores = []
        filter_batch_scores = []

        # compute the batch scores
        print(f"Batch {i} of {len(dataloader)}")

        for j, (x, y) in enumerate(zip(*batch)):
            print(f"Sample {j} of {len(batch[0])}")

            x = x.to(device)
            y = y.to(device)

            # computing the wavelet transform for the target sequence
            if len(x.shape) == 3:
                sequence_length = x.shape[0]
                signal = x[sequence_length //2][0] # assuming one channel
            else: 
                signal = x[0]

            # create filterbank assuming 1 channel
            filterbank.apply_dwt_filterbank(signal)
            
            # get the attribution mask
            mask, loss = mask_opt.fit(x, verbose=verbose)

            losses.append(loss)
            mask = mask.squeeze().cpu().detach().numpy() # shape (time, n_filters)

            # normalize 
            imp = torch.tensor(filterbank.get_filter_response(mask)) # shape (channels, time, n_filters)

            batch_scores.append(imp)
            filter_batch_scores.append(mask)

        # store the data
        masks.append(torch.stack(batch_scores)) # shape (batch_len, channels, time, n_filters)
        scores.append(np.stack(filter_batch_scores))  # shape (batch_len, time, n_filters)
        
    return masks, scores
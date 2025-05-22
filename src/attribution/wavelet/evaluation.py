import torch
import numpy as np
from src.attribution.wavelet import WaveletFilterbank, WaveletMask

def compute_wavelet_attribution( model, 
                                 dataloader, 
                                 filterbank_params = {'wavelet': 'db', 'w_len': 1, 'fs': 100, 'level': 5}, 
                                 device = 'cpu', 
                                 verbose = True,
                                 normalize = False,
                                 rescale = False):
    # define mask
    masks = []
    scores = []

    # create filterbank
    filterbank = WaveletFilterbank(**filterbank_params)

    # create FLEXtime mask
    mask_opt = WaveletMask(model, filterbank, device=device) 

    for i, batch in enumerate(dataloader):
        batch_scores = []
        filter_batch_scores = []

        # compute the batch scores
        print(f"Batch {i} of {len(dataloader)}")

        for j, (x, y) in enumerate(zip(*batch)):
            print(f"Sample {j} of {len(batch[0])}")

            x = x.to(device)
            y = y.to(device)

            # assuming one channel
            singal = x[0]

            # create filterbank assuming 1 channel
            filterbank.apply_dwt_filterbank(singal)
            
            # get the attribution mask
            mask, loss = mask_opt.fit(x, verbose=verbose, normalize=normalize, rescale=rescale)
            # print(type(mask))
            mask = mask.squeeze().cpu().detach().numpy() # shape (time, n_filters)
            # print(f"Mask shape: {mask.shape}")

            # normalize 
            imp = torch.tensor(filterbank.get_filter_response(mask)) # shape (channels, time, n_filters)
            # print(f"Imp shape: {imp.shape}")

            batch_scores.append(imp)
            filter_batch_scores.append(mask)

        # store the data
        masks.append(torch.stack(batch_scores)) # shape (batch_len, channels, time, n_filters)
        scores.append(np.stack(filter_batch_scores))  # shape (batch_len, time, n_filters)

        break

    return masks, scores
import torch
import numpy as np
from src.attribution.flextime import Filterbank, FLEXtimeMask

def compute_flextime_attribution(model, 
                                 dataloader, 
                                 filterbank_params = {'n_taps': 501, 'n_filters': 64, 'sample_freq': 8000, 'time_len': 1}, 
                                 device = 'cpu', 
                                 verbose = True):
    # define mask
    masks = []
    scores = []

    # create filterbank
    filterbank = Filterbank(**filterbank_params)
    
    # create FLEXtime mask
    mask_opt = FLEXtimeMask(model, filterbank, device=device)    

    for i, batch in enumerate(dataloader):
        batch_scores = []
        filter_batch_scores = []

        # compute the batch scores
        print(f"Batch {i} of {len(dataloader)}")

        for j, (x, y) in enumerate(zip(*batch)):
            print(f"Sample {j} of {len(batch[0])}")

            x = x.to(device)
            y = y.to(device)
            
            # get the attribution mask
            mask, loss = mask_opt.fit(x, verbose=verbose)
            mask = mask.squeeze().cpu().detach().numpy() # shape (n_filters, )

            # normalize 
            imp = torch.tensor(filterbank.get_filter_response(mask)) # shape (N//2+1, 1)

            batch_scores.append(imp)
            filter_batch_scores.append(mask)


        # store the data
        masks.append(torch.stack(batch_scores)) # shape (batch_len, N//2+1, 1)
        scores.append(np.stack(filter_batch_scores))  # shape (batch_len, n_filters)
        
    return masks, scores
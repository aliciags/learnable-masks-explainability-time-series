import torch
import numpy as np
from src.attribution.flextime import Filterbank, FLEXtimeMask

def compute_flextime_attribution(model, 
                                 data, 
                                 filterbank_params = {'n_taps': 501, 'n_filters': 64, 'smaple_freq': 8000}, 
                                 device = 'cpu'):
    # define mask
    mask = []

    
    # create filterbank
    filterbank = Filterbank(**filterbank_params)
    
    # create FLEXtime mask
    mask_opt = FLEXtimeMask(model, filterbank, device=device)
    mask.fit(data)
    return mask.get_mask()
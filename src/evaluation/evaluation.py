import pywt
import torch
import math
import numpy as np
import torch.nn.functional as F

from src.utils.sampling import upsampling_wavedec, downsample_wavedec, split_string

def evaluate_attributions(model, 
                          loader, 
                          attribution, 
                          quantiles, 
                          mode = 'insertion', 
                          domain = 'fft',
                          wavelet = 'db1', 
                          device = 'cpu',
                          level = None):
    """
    Evaluation function used to assess how a model's predictions change when 
    certain parts of the input are masked—either deleted or inserted—based on 
    their importance.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be evaluated.
    loader : torch.utils.data.DataLoader
        The test loader.
    attribution : list
        The attribution scores.
    quantiles : list
        The quantiles to be evaluated.
    mode : str
        The mode of evaluation. Can be 'insertion' or 'deletion'.
    domain : str
        The domain of the data. Can be 'fft', 'wavelet' or 'time'.
    device : str
        The device to be used for evaluation.

    Returns
    -------
    accuracies : list
        The accuracies of the model for each quantile.
    mean_true_class_probs : list
        The mean true class probabilities of the model for each quantile.
    ce_losses : list
        The cross entropy losses of the model for each quantile.
    """
    
    model.eval().to(device)

    if domain == 'wavelet':
        wavelet_name, filter_length = split_string(wavelet)
        filter_length = int(filter_length)

    with torch.no_grad():
        accuracies = []
        mean_true_class_probs = []
        ce_losses = []
        for quantile in quantiles:
            correct = 0
            total = 0
            ce_loss = 0
            mean_true_class_prob = 0

            for i, batch in enumerate(loader):
                # assuming the batch is a tuple of samples and true labels
                x, y = batch
                # assuming the time dimension is the last dimension
                time_dim = -1
                time_len = len(x[0][0])

                if domain == 'fft':
                    data = torch.fft.rfft(x, dim=time_dim).to(device)
                elif domain == 'wavelet':

                    # assuming one channel
                    wavelet_transform = []
                    coeffs = pywt.wavedec(x, wavelet, level=level)

                    for j in range(len(coeffs)):  # iterate over levels
                        level_list = []
                        for k in range(len(coeffs[j])):  # batch size
                            batch_list = []
                            for l in range(len(coeffs[j][k])):  # channels
                                signal = coeffs[j][k][l]
                                factor = math.ceil(time_len / len(signal))
                                upsampled = np.repeat(signal, factor)[:time_len]
                                batch_list.append(upsampled)
                            level_list.append(batch_list)
                        wavelet_transform.append(level_list)

                    wavelet_transform = np.moveaxis(np.array(wavelet_transform), 0, -1)
                    data = torch.tensor(wavelet_transform).float().to(device)

                    # print(f"Data shape: {data.shape}")

                elif domain == 'time':
                    data = x.to(device)
                
                shape = data.shape
                    
                # print(f"Attribution shape: {attribution[i].shape}")
                imp = attribution[i].reshape(shape).to(torch.float32).to(device)

                # flatten data and compute quantile
                flattened_imp = imp.reshape(shape[0], -1)
                k = int(quantile * flattened_imp.size(1))

                # find top k indices
                _, indices = torch.topk(flattened_imp, k=k, dim=1)

                # create mask
                mask = torch.zeros_like(flattened_imp, dtype=torch.bool)
                mask.scatter_(1, indices, True)
                mask = mask.view(shape).to(device)

                # apply mask to data
                if mode == 'insertion':
                    masked_data = data * mask
                elif mode == 'deletion':
                    masked_data = data * (~mask)

                else:
                    raise ValueError("mode must be 'insertion' or 'deletion'")

                # print(f"Masked data shape: {masked_data.shape}")
                

                if domain == 'fft':
                    data = torch.fft.irfft(masked_data, dim=time_dim).to(device)
                elif domain == 'wavelet':
                    # downsample the data to the original size
                    masked_data_np = masked_data.detach().cpu().numpy()
                    # print(f"Masked data shape numpy: {masked_data_np.shape}")

                    masked_data_np = np.moveaxis(masked_data_np, -2, -1)  # now shape: [batch_size, channels, levels, time]
                    # print(f"Masked data shape numpy after move axis: {masked_data_np.shape}")

                    reconstructed_data = []

                    for j in range(masked_data_np.shape[0]):  # over batch
                        for k in range(masked_data_np.shape[1]):  # over channels
                            coeffs = downsample_wavedec(time_len, masked_data_np[j][k], wavelet_name, filter_length)  # should be a flat list of np arrays
                            recon = pywt.waverec(coeffs, wavelet)
                            reconstructed_data.append(recon)

                    data = torch.tensor(np.array(reconstructed_data)).float().to(device)


                output = model(data.float()).detach().cpu()

                _, predicted = torch.max(output, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
                ce_loss += F.cross_entropy(output, y).item()/len(batch)
                mean_true_class_prob += torch.take_along_dim(F.softmax(output, dim=1), y.unsqueeze(1), dim = 1).sum().item()

                if len(attribution) != len(loader):
                    break

            accuracies.append(correct / total)
            ce_losses.append(ce_loss)
            mean_true_class_probs.append(mean_true_class_prob / total)

        return accuracies, mean_true_class_probs, ce_losses
import torch
import torch.nn.functional as F

def evaluate_attributions(model, loader, attribution, quantiles, mode = 'deletion', device = 'cpu'):
    """
    Evaluate the model using the given attribution and data.

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
                x, y = batch
                time_dim = -1

                data = torch.fft.rfft(x, dim=time_dim).to(device)
                shape = data.shape

                # if attribution == 'random':
                #     imp = torch.rand_like(data).float()
                # elif attribution == 'amplitude':
                #     imp = torch.abs(data)
                # else:
                    
                imp = attribution[i].reshape(shape).to(device)

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

                data = torch.fft.irfft(masked_data, dim=time_dim).to(device)
                output = model(data.float()).detach().cpu()

                _, predicted = torch.max(output, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
                ce_loss += F.cross_entropy(output, y).item()/len(batch)
                mean_true_class_prob += torch.take_along_dim(F.softmax(output, dim=1), y.unsqueeze(1), dim = 1).sum().item()

            accuracies.append(correct / total)
            ce_losses.append(ce_loss)
            mean_true_class_probs.append(mean_true_class_prob / total)

        return accuracies, mean_true_class_probs, ce_losses
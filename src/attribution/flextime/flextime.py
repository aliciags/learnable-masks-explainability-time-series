import torch
from src.attribution.flextime.filterbank import Filterbank

class FLEXtimeMask():
    """
    FLEXtimeMask is a class that generates a mask for FLEXtime.
    """

    def __init__(self, model, filterbank: Filterbank, regularization: str = 'l1', device: str = 'cpu'):
        self.model = model.to(device)
        self.filterbank = filterbank
        self.regularization = regularization
        self.device = device
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def fit(self, data, use_only_max: bool = True):
        self.model.eval()

        # to disable gradient computation for save memory and speed up process
        with torch.no_grad():
            # target tensor with class logits
            target = self.model(data.float().to(self.device))
            # softmax to convert logits into probabilities
            target = torch.nn.functional.softmax(target, dim=1)

            # set all other than max class to zero
            if use_only_max:
                # instead of probs target is class indices for each sample of max prob
                target = torch.argmax(target, dim=1)

        # initialize mask
        mask_shape = torch.tensor(data.shape)
        mask_shape[-1] = 1
        mask = torch.ones((*mask_shape, self.filterbank.n_filters), device=self.device)
        mask.requires_grad = True

        # optimizer
        optimizer = torch.optim.Adam([mask], lr=0.01)

        # precompute filterbank output
        data_bands = self.filterbank.apply_filterbank(data.cpu().numpy())
        data_bands = torch.tensor(data_bands).float().to(self.device).reshape(*data.shape, self.filterbank.n_filters)

        # training loop
        for epoch in range(100):
            optimizer.zero_grad()

            # apply mask and sum over filter dimension
            masked_data = (data_bands * mask).sum(-1)

            # forward pass
            output = self.model(masked_data)
            output = torch.nn.functional.softmax(output, dim=1)

            # calculate loss
            loss = self.loss_fn(output, target)

            # regularization assuming lambda = 1 in total loss
            if self.regularization == 'l1':
                loss += mask.abs().sum()
            elif self.regularization == 'l2':
                loss += mask.pow(2).sum()

            # backward pass
            loss.backward()
            optimizer.step()

            # clip mask values to [0, 1]
            mask.data = torch.clamp(mask, 0, 1)

            if epoch % 10 == 0:
                print(f'Epoch {epoch} Loss {loss.item()}')

        return mask

            

        


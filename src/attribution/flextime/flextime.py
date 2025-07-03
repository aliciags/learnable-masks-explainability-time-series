import torch
import numpy as np
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
        self.nfilters = filterbank.n_filters

    # def __init__(self, model, filterbank: Filterbank, regularization: str = 'l1', device: str = 'cpu'):
    #     self.model = model.to(device)
    #     self.filterbank = filterbank
    #     self.regularization = regularization
    #     self.device = device
    #     self.loss_fn = torch.nn.CrossEntropyLoss()

#     def fit(self, data, stopping: float = 1.0e-6, use_only_max: bool = True):
#         self.model.eval()
#         early_stopping_counter = 0

#         # data = data.unsqueeze(0)

#         # to disable gradient computation for save memory and speed up process
#         with torch.no_grad():
#             # target tensor with class logits
#             target = self.model(data.float().to(self.device))
#             # softmax to convert logits into probabilities
#             target = torch.nn.functional.softmax(target, dim=1)

#             # set all other than max class to zero
#             if use_only_max:
#                 # instead of probs target is class indices for each sample of max prob
#                 target = torch.argmax(target, dim=1)
            

#         # initialize mask to 0.5
#         mask_shape = torch.tensor(data.shape)
#         mask_shape[-1] = 1
#         mask = 0.5*torch.ones((*mask_shape, self.filterbank.n_filters), device=self.device)
#         mask.requires_grad = True

#         # optimizer
#         optimizer = torch.optim.Adam([mask], lr=0.01)

#         # precompute filterbank output
#         data_bands = self.filterbank.apply_filterbank(data.cpu().numpy())
#         data_bands = torch.tensor(data_bands).float().to(self.device).reshape(*data.shape, self.filterbank.n_filters)
#         # shape (1, time_len, n_filters)
#         # print(f"Data bands shape {data_bands.shape}")


#         l = float('inf')

#         # training loop
#         for epoch in range(500):
#             optimizer.zero_grad()

#             # apply mask and sum over filter dimension
#             masked_data = (data_bands * mask).sum(-1)

#             # forward pass
#             output = self.model(masked_data)
#             output = torch.nn.functional.softmax(output, dim=1)

#             # calculate loss
#             loss = self.loss_fn(output, target)

#             # regularization assuming lambda = 1 in total loss
#             if self.regularization == 'l1':
#                 loss += mask.abs().sum()
#             elif self.regularization == 'l2':
#                 loss += mask.pow(2).sum()

#             # backward pass
#             loss.backward()
#             optimizer.step()

#             # clip mask values to [0, 1]
#             mask.data = torch.clamp(mask, 0, 1)

#             if epoch % 10 == 0:
#                 print(f'Epoch {epoch} Loss {loss.item()}')

#             if stopping is not None:
#                 if abs(l - loss.item()) < stopping:
#                     early_stopping_counter += 1
#                 l = loss.item()
#                 if early_stopping_counter > 10:
#                     break


#         return mask


    def fit(self,
            data,
            time_dim: int = -1,
            n_epoch: int = 250,
            learning_rate: float = 1e-2,
            keep_ratio: float = 0.01,
            reg_factor_init: float = 1.0,
            reg_factor_dilation: float = 1.0,
            time_reg_strength: float = 0.0,
            stopping: float = 1e-5,
            use_only_max: bool = True,
            verbose: bool = True,
            patience: int = 10):

        self.model.eval()
        early_stopping_counter = 0

        if len(data.shape) == 3:
            data = data.unsqueeze(0)
        data = data.float().to(self.device)

        with torch.no_grad():
            target = self.model(data)
            target = torch.nn.functional.softmax(target, dim=1)
            if use_only_max:
                target = torch.argmax(target, dim=1)

        # Initialize mask
        mask_shape = torch.tensor(data.shape)
        mask_shape[time_dim] = 1
        mask = (0.1 * torch.ones((*mask_shape, self.nfilters), device=self.device)).detach()
        mask.requires_grad_()

        optimizer = torch.optim.Adam([mask], lr=learning_rate)

        # Optional regularization target
        if self.regularization == 'ratio':
            reg_ref = torch.zeros(int((1 - keep_ratio) * self.nfilters))
            reg_ref = torch.cat((reg_ref, torch.ones(self.nfilters - reg_ref.shape[0]))).to(self.device)

        # Precompute filtered bands
        data_bands = self.filterbank.apply_filterbank(data.cpu().numpy())
        data_bands = torch.tensor(data_bands).float().to(self.device).reshape(*data.shape, self.nfilters)

        reg_strength = reg_factor_init
        reg_multiplicator = np.exp(np.log(reg_factor_dilation) / max(n_epoch, 1))
        
        prev_loss = float('inf')
        total_loss_list = []

        for epoch in range(n_epoch):
            optimizer.zero_grad()

            masked_data = (data_bands * mask).sum(-1)
            output = self.model(masked_data)
            output = torch.nn.functional.softmax(output, dim=1)
            target_loss = self.loss_fn(output, target)

            # Compute regularization
            if self.regularization == 'l1':
                reg_loss = torch.max(mask.abs().mean() - keep_ratio, torch.tensor(0., device=self.device))
            elif self.regularization == 'l2':
                reg_loss = mask.pow(2).mean()
            elif self.regularization == 'ratio':
                reg_loss = ((torch.sort(mask)[0] - reg_ref)**2).mean()
            else:
                reg_loss = 0.0

            # Time smoothness regularization
            if epoch < 10:
                time_strength = 0.0
            else:
                time_strength = time_reg_strength
            time_reg = (mask[..., 1:] - mask[..., :-1]).abs().mean() if time_strength > 1e-6 else 0.0

            # Total loss
            total_loss = target_loss + reg_strength * reg_loss + time_strength * time_reg
            total_loss.backward()
            optimizer.step()
            mask.data = torch.clamp(mask, 0, 1)

            reg_strength *= reg_multiplicator
            total_loss_list.append(total_loss.item())

            if verbose and epoch % 10 == 0:
                print(f"[Epoch {epoch}] Loss: {total_loss.item():.4f}, Target: {target_loss.item():.6f}, Reg: {reg_loss:.6f}, L1: {mask.abs().mean():.4f}")

            # Early stopping
            if abs(prev_loss - total_loss.item()) < stopping:
                early_stopping_counter += 1
            else:
                early_stopping_counter = 0
            prev_loss = total_loss.item()
            if early_stopping_counter > patience:
                break

        return mask, total_loss_list

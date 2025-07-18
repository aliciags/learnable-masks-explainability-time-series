import torch
import numpy as np
from src.attribution.wavelet.wavelet import WaveletFilterbank
from src.utils.sampling import downsample_wavedec, upsampling_wavedec

class WaveletMask:
    def __init__(self, model, wavelet_filterbank: WaveletFilterbank, regularization='l1', device='cpu'):
        self.model = model.to(device)
        self.filterbank = wavelet_filterbank
        self.device = device
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.regularization = regularization

    def fit(self,
            data,
            n_epoch: int = 250,
            learning_rate: float = 1.0e-2,
            keep_ratio: float = 0.01,
            reg_factor_init: float = 1.0,
            reg_factor_dilation: float = 1.0,
            stopping: float = 1.0e-5,
            patience: int = 10,
            verbose: bool = True,
            normalize: bool = False,
            rescale: bool = False,
            use_only_max: bool = True):
        
        self.model.eval()
        early_stopping_counter = 0

        if len(data.shape) == 3:
            data = data.unsqueeze(0)
        data = data.float().to(self.device)

        # Get model target
        with torch.no_grad():
            target = self.model(data)
            target = torch.nn.functional.softmax(target, dim=1)
            if use_only_max:
                target = torch.argmax(target, dim=1)

        # Initialize mask
        mask_shape = torch.tensor(data.shape)
        mask = (0.1 * torch.ones((*mask_shape, self.filterbank.nbanks), device=self.device)).detach()
        mask.requires_grad_()

        optimizer = torch.optim.Adam([mask], lr=learning_rate)

        # Regularization
        if self.regularization == 'ratio':
            reg_ref = torch.zeros(int((1 - keep_ratio) * self.filterbank.nbanks))
            reg_ref = torch.cat((reg_ref, torch.ones(self.filterbank.nbanks - reg_ref.shape[0]))).to(self.device)

        # Get filtered bands from wavelet filterbank
        bands = self.filterbank.get_wavelet_bands()
        bands = torch.tensor(bands).float().to(self.device)
        bands = bands.permute(-1, -2)
        bands = bands.unsqueeze(0)  # (n_channels, time, n_filters)

        reg_strength = reg_factor_init
        reg_multiplicator = np.exp(np.log(reg_factor_dilation) / max(n_epoch, 1))

        prev_loss = float('inf')
        total_loss = []

        for epoch in range(n_epoch):
            optimizer.zero_grad()
            # Apply mask and sum
            masked = (bands * mask).sum(-1)

            # # move last axis to the first
            # masked = masked.squeeze()
            # masked = masked.permute(1, 0)
            # downsample = downsample_wavedec(len(masked[0]), masked.detach().cpu().numpy(), self.filterbank.wavelet, 1)
            # s = pywt.waverec(downsample, self.filterbank.wavelet)
            # # make it a torch tensor
            # masked = torch.tensor(s).float().to(self.device)
            # masked = masked.unsqueeze(0)
            
            output = self.model(masked)
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

            loss = target_loss + reg_strength * reg_loss
            loss.backward()
            optimizer.step()

            # Clamp mask to [0, 1]
            mask.data = torch.clamp(mask, 0, 1)
            total_loss.append(loss.item())

            reg_strength *= reg_multiplicator

            if verbose and epoch % 10 == 0:
                print(f'Epoch {epoch}: Loss={loss.item():.4f}, Target={target_loss.item():.4f}, Reg={reg_loss.item():.4f}')

            # Early stopping
            if abs(prev_loss - total_loss[-1]) < stopping:
                early_stopping_counter += 1
            else:
                early_stopping_counter = 0
            prev_loss = total_loss[-1]
            if early_stopping_counter > patience:
                break

        return mask, total_loss


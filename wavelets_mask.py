import os
import pywt
import torch
import random
import pickle
import numpy as np
from types import SimpleNamespace
from quantus.metrics import Complexity
from torch.utils.data import DataLoader, TensorDataset

from src.utils import split_string
from src.models.simple import SimpleCNN
from src.attribution import compute_attribution
from src.evaluation.evaluation import evaluate_attributions

def set_seed(seed: int = 42):
    random.seed(seed)                          # Python built-in random
    np.random.seed(seed)                       # NumPy random
    torch.manual_seed(seed)                    # CPU random seed
    torch.cuda.manual_seed(seed)               # GPU random seed (if used)
    torch.cuda.manual_seed_all(seed)           # All GPUs (if multiple GPUs)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Optional: for hash-based operations
    os.environ['PYTHONHASHSEED'] = str(seed)

# set seed
set_seed(42)

# to CUDA if available otherwise to CPU
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
    
print(f"Using device: {device}")

# load the model
model = SimpleCNN(in_channels=1, out_channels=2, hidden_size=64, kernel_size=5)

# load the model weights
model.load_state_dict(torch.load("./model/checkpoint/simpleCNN_1.pth"))
model.to(device)

# Load and shape synthetic test data
x = None
y = None

# load the data from synthetic data
data_folder = "./data/synthetic/test_1"
data_files = os.listdir(data_folder)
for file in data_files:
    if "samples_0" in file:
        if x is None and y is None:
            x = np.load(os.path.join(data_folder, file))
            y = np.zeros(5000)
        else:
            x = np.concatenate([x, np.load(os.path.join(data_folder, file))])
            y = np.concatenate([y, np.zeros(5000)])
    elif "samples_1" in file:
        if x is None and y is None:
            x = np.load(os.path.join(data_folder, file))
            y = np.ones(5000)
        else:
            x = np.concatenate([x, np.load(os.path.join(data_folder, file))])
            y = np.concatenate([y, np.ones(5000)])
    else:
        print("File not recognized")
        continue
    
# simualte one channel
x = x[:, np.newaxis, :]

print(x.shape)
print(y.shape)

# convert the data to torch tensors
x = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# Shuffle indices once
indices = torch.randperm(len(x))

# Apply the shuffle
x_shuffled = x[indices]
y_shuffled = y[indices]

# create the dataset
dataset = TensorDataset(x_shuffled, y_shuffled)

# create the DataLoader
test_loader = DataLoader(dataset, batch_size=128)

# read data
# folder = 'public/simple/'
# path = f'{folder}wavelets_results.pkl'

# with open(path, 'rb') as f:
#     attributions = pickle.load(f)
# print(f"Attributions loaded from {path}")

comp = Complexity()

# define metrics
attributions = {}
attributions['deletion'] = {}
attributions['insertion'] = {}
predictions = []
labels = []
complexities = {}
grad_complexties = {}

# prediction
for batch in test_loader:
    # get data
    x, y = batch
    x = x.to(device)
    output = model(x)

    predictions.append(output)
    labels.append(y)

# concatenat the predictions and the labels through the first dim
predictions = torch.cat(predictions, dim=0)
labels = torch.cat(labels, dim=0)

# save it in the attributions dict
attributions['predictions'] = predictions
attributions['labels'] = labels

# defining parameters
fs = 1000
batch_size = 128
method = 'wavelet'
quantiles = np.arange(0, 1.05, 0.05)

# define all the wavelets to test
wavelets =  ['db1', 'db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8', 'db9', 'db10', 
             'sym2', 'sym3', 'sym4', 'sym5', 'sym6', 'sym7', 'sym8', 'sym9', 'sym10',
             'coif1', 'coif2', 'coif3', 'coif4', 'coif5']

for w in wavelets:
    wavelet, w_len = split_string(w)
    w_len = int(w_len)
    level = 5 # pywt.dwt_max_level(fs, w)
    key_ = f'wavelet_{wavelet}{w_len}_{level}_{batch_size}'
    print(key_)

    # compute the attributions
    args = SimpleNamespace(wavelet=wavelet, len_w=w_len, level=level, sample_freq=fs)
    attrs, masks = compute_attribution(method = method, model = model, test_loader= test_loader, args = args, device=device)

    attributions[key_] = attrs
    attributions[f'filtermasks_{key_}'] = masks

    for mode in ['deletion', 'insertion']:
        if not mode in attributions.keys():
            attributions[mode] = {}
        
        acc_scores = evaluate_attributions(model, test_loader, attributions[key_], quantiles=quantiles, mode=mode, device=device, domain='wavelet', wavelet=w, level=level)
        attributions[mode][key_] = acc_scores

    if not key_ in complexities.keys():
        complexities[key_] = []
        grad_complexties[key_] = []

    scores = []
    grad_scores = []

    for i in range(len(attributions[key_])):
        expl = np.reshape(attributions[key_][i], (attributions[key_][i].shape[0], -1))
        expl = expl.to(dtype=torch.float32).numpy()
        
        ex = np.maximum(attributions[key_][i].numpy(), 0)
        if 'filterbank' in key_:
            ex = np.transpose(ex, (0, 2, 1))

        # min max normalize
        ex_min = np.min(ex, axis = -1, keepdims=True)
        ex_max = np.max(ex, axis = -1, keepdims=True)
        ex = (ex - ex_min) / (ex_max - ex_min + 1e-10)
        
        expl_grad = np.abs(np.diff(ex, axis = -1)).sum(axis=-1)
        expl_grad = np.reshape(expl_grad, (attributions[key_][i].shape[0], -1))

        expl = np.maximum(expl, 0)
        # check if all expl values are zero
        if np.all(expl == 0):
            print("All zeros")
            # add a small epsilon to avoid division by zero
            expl = np.ones_like(expl) * 1e-10

        # to compute complexities it has to be a numpy float32 otherwise err
        complexity = comp.evaluate_batch(expl, expl)
        complexity = np.nan_to_num(complexity)
        expl_grad = np.nan_to_num(expl_grad)
        scores += complexity.tolist()
        grad_scores += list(expl_grad)

    complexities[key_].append(np.mean(scores))
    grad_complexties[key_].append(np.mean(grad_scores))

print(complexities)
print(grad_complexties)

attributions['complexities'] = complexities
attributions['grad_complexities'] = grad_complexties

# dump to file
folder = 'public/simple/'
path = f'{folder}_wavelets_results_5levels.pkl'

with open(path, 'wb') as f:
    pickle.dump(attributions, f)
print(f"Saved to {path}")
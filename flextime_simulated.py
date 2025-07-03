import os
import torch
import pickle
import numpy as np
import multiprocessing

from src.models.simple import SimpleCNN
from torch.utils.data import TensorDataset, DataLoader

from types import SimpleNamespace
from src.attribution import compute_attribution
from src.evaluation.evaluation import evaluate_attributions
from quantus.metrics import Complexity


if __name__ == '__main__':
    multiprocessing.freeze_support()

    def set_seed(seed: int = 42):
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

    # to mps if available otherwise to cpu
    device = torch.device('mps' if torch.backends.mps.is_available()  else 'cpu')
    print(f"Using device: {device}")

    # load the model
    model = SimpleCNN(in_channels=1, out_channels=2, hidden_size=64, kernel_size=5)

    # load the model weights
    model.load_state_dict(torch.load("./model/checkpoint/simpleCNN_4.pth", map_location=torch.device('mps')))
    model.to(device)

    x = None
    y = None

    # load the data from synthetic data
    data_folder = "./data/synthetic/test_4"
    data_files = os.listdir(data_folder)
    for file in data_files:
        if "samples_0_0" in file:
            if x is None and y is None:
                x = np.load(os.path.join(data_folder, file))
                y = np.zeros(5000)
            else:
                x = np.concatenate([x, np.load(os.path.join(data_folder, file))])
                y = np.concatenate([y, np.zeros(5000)])
        elif "samples_1_0" in file:
            if x is None and y is None:
                x = np.load(os.path.join(data_folder, file))
                y = np.ones(5000)
            else:
                x = np.concatenate([x, np.load(os.path.join(data_folder, file))])
                y = np.concatenate([y, np.ones(5000)])
        else:
            print("File not recognized")
            continue
        
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

    # Initialize metrics
    attributions = {
        'deletion': {},
        'insertion': {}
    }
    predictions = []
    labels = []
    complexities = {}
    grad_complexities = {}

    # Process data in smaller chunks
    chunk_size = 16  # Process 16 samples at a time
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # Split batch into smaller chunks
            for i in range(0, len(batch[0]), chunk_size):
                # Get chunk of data
                x_chunk = batch[0][i:i+chunk_size].to(device)
                y_chunk = batch[1][i:i+chunk_size]

                # Forward pass
                # with torch.cuda.amp.autocast():  # Use mixed precision if available
                output = model(x_chunk)

                # Move predictions to CPU and store
                predictions.append(output.cpu())
                labels.append(y_chunk)

                # Clear GPU memory
                del x_chunk, output
                # torch.cuda.empty_cache()

            print(f"Processed batch {batch_idx+1}/{len(test_loader)}")
            
            # Clear batch variables
            del batch
            # torch.cuda.empty_cache()

    # concatenat the predictions and the labels through the first dim
    predictions = torch.cat(predictions, dim=0)
    labels = torch.cat(labels, dim=0)

    # save it in the attributions dict
    attributions['predictions'] = predictions
    attributions['labels'] = labels

    key_ = 'flextime_128'

    # compute attribution of the masks
    args = SimpleNamespace(n_taps=501, n_filters=64, sample_freq=1000, time_len=1000)
    attrs, masks = compute_attribution(method='flextime', model=model, test_loader=test_loader, device=device, args=args)

    attributions[key_] = attrs
    attributions[f'filtermasks_{key_}'] = masks

    comp = Complexity()

    quantiles = np.arange(0, 1.05, 0.05)


    for mode in ['deletion', 'insertion']:
        if not mode in attributions.keys():
            attributions[mode] = {}
        
        acc_scores = evaluate_attributions(model, test_loader, attributions[key_], quantiles=quantiles, mode=mode, device=device, domain='fft')
        attributions[mode][key_] = acc_scores

    
    # dump to file
    folder = 'public/simple/'
    path = f'{folder}{key_}.pkl'

    # save the masks
    with open(path, "wb") as f:
        pickle.dump(attributions, f)
    print(f"Saved to {path}")

    if not key_ in complexities.keys():
        complexities[key_] = []
        grad_complexities[key_] = []

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
    grad_complexities[key_].append(np.mean(grad_scores))

    print(complexities)
    print(grad_complexities)

    attributions['complexities'] = complexities
    attributions['grad_complexities'] = grad_complexities

    # dump to file
    folder = 'public/simple/'
    path = f'{folder}{key_}.pkl'

    # save the masks
    with open(path, "wb") as f:
        pickle.dump(attributions, f)
    print(f"Saved to {path}")



import os
import yaml
import torch
import pickle
import importlib
import multiprocessing

from types import SimpleNamespace
from src.attribution import compute_attribution

from physioex.physioex.data import PhysioExDataModule
from physioex.physioex.train.models.load import load_model

target_package = "physioex.train.networks.utils.target_transform"
target_class = "get_mid_label"
target = getattr(importlib.import_module(target_package), target_class)

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

    target_package = "physioex.train.networks.utils.target_transform"
    target_class = "get_mid_label"
    target = getattr(importlib.import_module(target_package), target_class)

    # Optimized datamodule configuration with reduced batch size
    datamodule = PhysioExDataModule(
        datasets=["sleepedf"],     # list of datasets to be used
        batch_size=64,            # Reduced batch size for better memory usage
        preprocessing="raw",       # preprocessing method
        selected_channels=["EEG"], # channels to be selected
        sequence_length=7,         # length of the sequence
        target_transform= target,  # since seq to epoch, target seq
        num_workers = 4,          # Reduced number of workers
        data_folder = "data"    # path to the data folder
    )

    # get the test DataLoaders
    test_loader = datamodule.test_dataloader()

    print(f"Number of batches: {len(test_loader)}")
    print(f"Batch size: {test_loader.batch_size}")
    print(f"Total samples: {len(test_loader) * test_loader.batch_size}")

    checkpoint_path = "./model/checkpoint/"

    with open("./config.yaml", "r") as file:
        config = yaml.safe_load(file)

    network_config = config["model_config"]

    # load the model class
    model_package, model_class = config["model"].split(":")
    model_class = getattr(importlib.import_module(model_package), model_class)

    # casting the float types
    network_config['learning_rate'] = float(network_config['learning_rate'])
    network_config['weight_decay'] = float(network_config['weight_decay'])
    network_config['adam_beta_1'] = float(network_config['adam_beta_1'])
    network_config['adam_beta_2'] = float(network_config['adam_beta_2'])
    network_config['adam_epsilon'] = float(network_config['adam_beta_2'])

    # load the model
    best_checkpoint = "fold=-1-epoch=19-step=19898-val_acc=0.83.ckpt"

    model = load_model(
        model = config["model"],
        model_kwargs = network_config,
        ckpt_path = checkpoint_path + best_checkpoint
    )

    # Move model to device
    model = model.to(device)
    model.eval()  # Set model to evaluation mode

    # Memory optimization
    # torch.cuda.empty_cache()
    # torch.backends.cudnn.benchmark = True

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

    key_ = 'flextime_7_64_64'

    # compute attribution of the masks
    args = SimpleNamespace(n_taps=501, n_filters=64, sample_freq=100, time_len=3000)
    attrs, masks = compute_attribution(method='flextime', model=model, test_loader=test_loader, device=device, args=args)

    attributions[key_] = attrs
    attributions[f'filtermasks_{key_}'] = masks

    folder = 'public/sleepedf/'
    path = f'{folder}{key_}.pkl'

    # save the masks
    with open(path, "wb") as f:
        pickle.dump(attributions, f)
    print(f"Saved to {path}")



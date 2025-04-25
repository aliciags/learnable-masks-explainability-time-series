import gc
import yaml
import torch
import pickle
import importlib
import numpy as np

from src.attribution.flextime import compute_flextime_attribution
from src.evaluation import evaluate_attributions

from physioex.physioex.data import PhysioExDataModule
from physioex.physioex.train.models.load import load_model

# clear cache
gc.collect()
torch.cuda.empty_cache()
torch.cuda.ipc_collect()
torch.cuda.reset_peak_memory_stats()

# define default
Fs = 100
time_dim = -1
time_length = 3000
batch_size = 32     # due to pytorch memory allocation limit
attributions = {}
attributions['deletion'] = {}
attributions['insertion'] = {}

# to cuda if available otherwise to cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# define target function
target_package = "physioex.train.networks.utils.target_transform"
target_class = "get_mid_label"
target = getattr(importlib.import_module(target_package), target_class)

# load test_dataset with shuffle Flase for sleepedf in hpc
datamodule = PhysioExDataModule(
    datasets=["sleepedf"],     # list of datasets to be used
    batch_size= batch_size,    # batch size for the DataLoader
    preprocessing="raw",       # preprocessing method
    selected_channels=["EEG"], # channels to be selected
    sequence_length=7,         # length of the sequence
    target_transform= target,  # since seq to epoch, target seq
    num_workers = 10,          # number of parallel workers
    data_folder = "/work3/s241931/data/"    # path to the data folder
)

# get the test DataLoaders
test_loader = datamodule.test_dataloader()

print(len(test_loader)) # number of batches in the test set

# load model for sleepedf in hpc
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
network_config['adam_epsilon'] = float(network_config['adam_epsilon'])

# load the model
best_checkpoint = "fold=-1-epoch=9-step=10170-val_acc=0.82-7.ckpt"

model = load_model(
    model = config["model"],
    model_kwargs = network_config,
    ckpt_path = checkpoint_path + best_checkpoint
)

predictions = []
labels = []

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

# evaluate flextime
n_filters = 128
numtaps = 501
key_ = f'flexime_{n_filters}_{numtaps}_32'

attribution, mask = compute_flextime_attribution(model, test_loader, {'n_taps': numtaps, 'n_filters': n_filters, 'sample_freq': Fs, 'time_len': time_length }, device=device)
attributions[key_] = attribution
attributions[f'filtermasks_{key_}'] = mask

# compute accuracy scores
quantiles = np.arange(0, 1.05, 0.05)
for mode in ['deletion', 'insertion']:
    if not mode in attributions.keys():
        attributions[mode] = {}
    
    acc_scores = evaluate_attributions(model, test_loader, attributions[key_], quantiles=quantiles, mode=mode, device=device)
    attributions[mode][key_] = acc_scores

# dump to file
folder = 'public/simple/'
path = f'{folder}{key_}.pkl'

with open(path, 'wb') as f:
    pickle.dump(attributions, f)
print(f"Saved to {path}")
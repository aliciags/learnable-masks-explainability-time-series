import yaml
import torch
import pickle
import importlib

from types import SimpleNamespace
from src.attribution import compute_attribution

from physioex.physioex.data import PhysioExDataModule
from physioex.physioex.train.models.load import load_model



target_package = "physioex.train.networks.utils.target_transform"
target_class = "get_mid_label"
target = getattr(importlib.import_module(target_package), target_class)

datamodule = PhysioExDataModule(
    datasets=["sleepedf"],     # list of datasets to be used
    batch_size=128,            # batch size for the DataLoader
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

args = SimpleNamespace()
args.n_taps = 501
args.n_filters = 64
args.sample_freq = 100
args.time_len = 3000 # 30s * 100Hz

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

attributions = {}
key_ = 'flextime_7_128_64'

# compute attribution of the masks
masks, scores = compute_attribution('flextime', model, test_loader, device=device, args=args)

attributions[key_] = masks
attributions[f'filtermasks_{key_}'] = scores

# save the masks
with open("public/masks_sleepEDF_7_64.pkl", "wb") as f:
    pickle.dump(attributions, f)

print(f"Saved to public/masks_sleepEDF_7_64.pkl")



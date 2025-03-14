
import os
import yaml
import importlib

from physioex.train.utils import train, test
from physioex.data import PhysioExDataModule
from physioex.train.models.load import load_model


target_package = "physioex.train.networks.utils.target_transform"
target_class = "get_mid_label"
target = getattr(importlib.import_module(target_package), target_class)

datamodule = PhysioExDataModule(
    datasets=["sleepedf"],     # list of datasets to be used
    batch_size=128,            # batch size for the DataLoader
    preprocessing="raw",       # preprocessing method
    selected_channels=["EEG"], # channels to be selected
    sequence_length=19,         # length of the sequence
    target_transform= target,  # since seq to epoch, target seq
    num_workers = 10,          # number of parallel workers
    data_folder = "/work3/s241931/data/"    # path to the data folder
)

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


# Train the model
best_checkpoint = train(
    datasets = datamodule,
    model_class = model_class,
    model_config = network_config,
    checkpoint_path = checkpoint_path,
    batch_size = 128,
    max_epochs = 10,
    resume = False
)

# best_checkpoint = "fold=-1-epoch=9-step=20214-val_acc=0.82.ckpt"

# Test the model
results_dataframe = test(
    datasets = datamodule,
    model_class = model_class,
    model_config = network_config,
    checkpoint_path = os.path.join( checkpoint_path, best_checkpoint ),
    batch_size = 128,
    results_path = checkpoint_path,
)

print(results_dataframe.head())
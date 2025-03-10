
import os
import yaml
import importlib

from physioex.train.utils import train, test
from physioex.data import PhysioExDataModule

datamodule = PhysioExDataModule(
    datasets=["sleepedf"],     # list of datasets to be used
    batch_size=64,             # batch size for the DataLoader
    preprocessing="raw",       # preprocessing method
    selected_channels=["EEG"], # channels to be selected
    sequence_length=21,        # length of the sequence
    data_folder = "/work3/s241931/data/"    # path to the data folder
)

checkpoint_path = "./model/checkpoint/"

# with open("./physioex/physioex/train/networks/config/chambon2018.yaml", "r") as file:
with open("./config.yaml", "r") as file:
    config = yaml.safe_load(file)

network_config = config["model_config"]

# load the loss function 
loss_package, loss_class = network_config["loss"].split(":")
model_loss = getattr(importlib.import_module(loss_package), loss_class)

# in case you provide model_name the system loads the additional model parameters from the library
if "model_name" in config:
    model_name = config["model_name"]

# load the model class
model_package, model_class = config["model"].split(":")
model_class = getattr(importlib.import_module(model_package), model_class)

datamodule_kwargs = {
    "selected_channels" : ["EEG"], # needs to match in_channels
    "sequence_length" : int(network_config["sequence_length"]),
    "target_transform" : config["target_transform"],
    "preprocessing" : config["input_transform"],
    "data_folder" : "/work3/s241931/data/",
}

# casting the float types
network_config['learning_rate'] = float(network_config['learning_rate'])
network_config['weight_decay'] = float(network_config['weight_decay'])
network_config['adam_beta_1'] = float(network_config['adam_beta_1'])
network_config['adam_beta_2'] = float(network_config['adam_beta_2'])
network_config['adam_epsilon'] = float(network_config['adam_epsilon'])


# Train the model
best_checkpoint = train(
    datasets = datamodule,
    datamodule_kwargs = datamodule_kwargs,
    model_class = model_class,
    model_config = network_config,
    checkpoint_path = checkpoint_path,
    batch_size = 64,
    max_epochs = 10
)

# Test the model
results_dataframe = test(
    datasets = datamodule,
    datamodule_kwargs = datamodule_kwargs,
    model_class = model_class,
    model_config = network_config,
    chekcpoint_path = os.path.join( checkpoint_path, best_checkpoint ),
    batch_size = 64,
    results_dir = checkpoint_path,  # if you want to save the test results 
                                    # in your checkpoint directory
)
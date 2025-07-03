import os
import yaml
import importlib
import multiprocessing

from physioex.physioex.train.utils import train, test
from physioex.physioex.data import PhysioExDataModule
from physioex.physioex.train.models.load import load_model

DATA_PATH = "./data"
CHECKPOINT_PATH = "./model/checkpoint/"
CONFIG_PATH = "./config.yaml"

BATCH_SIZE = 128

if __name__ == '__main__':
    multiprocessing.freeze_support()

    target_package = "physioex.train.networks.utils.target_transform"
    target_class = "get_mid_label"
    target = getattr(importlib.import_module(target_package), target_class)

    datamodule = PhysioExDataModule(
        datasets=["sleepedf"],     # list of datasets to be used
        batch_size=BATCH_SIZE,            # batch size for the DataLoader
        preprocessing="raw",       # preprocessing method
        selected_channels=["EEG"], # channels to be selected
        sequence_length=7,         # length of the sequence
        target_transform= target,  # since seq to epoch, target seq
        num_workers = 8,          # number of parallel workers
        data_folder = DATA_PATH    # path to the data folder
    )

    checkpoint_path = CHECKPOINT_PATH

    with open(CONFIG_PATH, "r") as file:
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
        batch_size = BATCH_SIZE,
        max_epochs = 20,
        resume = False
    )

    # Test the model
    results_dataframe = test(
        datasets = datamodule,
        model_class = model_class,
        model_config = network_config,
        checkpoint_path = os.path.join( checkpoint_path, best_checkpoint ),
        batch_size = BATCH_SIZE,
        results_path = checkpoint_path,
    )

    print(results_dataframe.head())
import os
from physioex.physioex.data import PhysioExData
from physioex.physioex.preprocess.sleepedf import SLEEPEDFPreprocessor
from physioex.physioex.preprocess.utils.signal import  xsleepnet_preprocessing
from physioex.physioex.data import PhysioExDataModule



class DataLoader:
    def __init__(self, data_folder: str, batch_size: int, num_workers: int):
        self.data_folder = data_folder
        self.batch_size = batch_size
        self.num_workers = num_workers

    def get_data_loader(self, dataset_name: str, preprocessing: str, selected_channels: list, sequence_length: int, target_transform: callable):
        if self.data_folder is None:
            raise ValueError("data_folder is not set")
        else:
            # read the folder
            with os.scandir(self.data_folder) as entries:
                if len(entries) == 0:
                    # The folder is empty load and preprocess data
                    self.preprocess_data()
                else:
                    for entry in entries:
                        if entry.is_dir() and entry.name == dataset_name:
                            datamodule = PhysioExDataModule(
                                datasets=[dataset_name],
                                batch_size=self.batch_size,
                                preprocessing=preprocessing,
                                selected_channels=selected_channels,
                                sequence_length=sequence_length,
                                target_transform= target_transform,
                                num_workers = self.num_workers,
                                data_folder = self.data_folder
                            )
                            return datamodule
                        else:
                            raise ValueError(f"dataset {dataset_name} is not found in the data folder")
    
    def preprocess_data(self):
        if self.dataset == 'sleepedf':
            preprocessor = SLEEPEDFPreprocessor(
                preprocessors_name = ["xsleepnet"], # the name of the preprocessor
                preprocessors = [xsleepnet_preprocessing], # the callable preprocessing method
                preprocessor_shape = [[3, 29, 129]], # the output of the signal after preprocessing, the first element depends on the number of channels
                data_folder = self.data_folder
            )
        elif self.dataset == 'simu':
            # simulate data
            pass
        else:
            raise ValueError(f"dataset {self.dataset} is not supported")

        preprocessor.run()  
    
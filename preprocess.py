from physioex.physioex.preprocess.sleepedf import SLEEPEDFPreprocessor  

# Define preprocessing functions, it's just a Callable method on each signal
from physioex.physioex.preprocess.utils.signal import  xsleepnet_preprocessing

# Initialize Preprocessor 
# I believe the shape is not needed, already defined in the class by default, only the folder data
# By default 3 channels not 4
preprocessor = SLEEPEDFPreprocessor(
    preprocessors_name = ["xsleepnet"], # the name of the preprocessor
    preprocessors = [xsleepnet_preprocessing], # the callable preprocessing method
    preprocessor_shape = [[3, 29, 129]], # the output of the signal after preprocessing, the first element depends on the number of channels
    data_folder = "./data"
)

# Run preprocessing
preprocessor.run()

from physioex.physioex.preprocess.sleepedf import SLEEPEDFPreprocessor  

# Define preprocessing functions, it's just a Callable method on each signal
from physioex.physioex.preprocess.utils.signal import  xsleepnet_preprocessing

# Initialize Preprocessor 
preprocessor = SLEEPEDFPreprocessor(
    preprocessors_name = ["xsleepnet"], # the name of the preprocessor
    preprocessors = [xsleepnet_preprocessing], # the callable preprocessing method
    preprocessor_shape = [[3, 29, 129]], # the output of the signal after preprocessing, the first element depends on the number of channels
    data_folder = "/work3/s241931/data/"
)

# Run preprocessing
preprocessor.run()

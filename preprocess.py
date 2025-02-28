from physioex.physioex.preprocess.sleepedf import SLEEPEDFPreprocessor  

# Define preprocessing functions, it's just a Callable method on each signal
from physioex.physioex.preprocess.utils.signal import  xsleepnet_preprocessing

# Initialize Preprocessor
preprocessor = SLEEPEDFPreprocessor(
    preprocessors_name = ["xsleepnet"], # the name of the preprocessor
    preprocessors = [xsleepnet_preprocessing], # the callable preprocessing method
    preprocessor_shape = [[4, 29, 129]], # the output of the signal after preprocessing, 
                                         # the first element (4) depends on the number of 
                                         # channels available in your system. In HMC they are 4.
    data_folder = "./data"
)

# Run preprocessing
preprocessor.run()

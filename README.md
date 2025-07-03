# Learnable masks for time series explainability

Firstly python 3.10 is required. Once the version is checked, the dependencies can be installed with:

```bash
pip install -r requirements.txt
```

Additionally PhysioEx is required. To install it, run the following command:

```bash
git clone https://github.com/guidogagl/physioex.git
cd physioex
pip install -e .
```

To generate synthetic data, run the following command:

```bash
python3 data.py
```

Otherwise to donwload the SleepEDF dataset, run the following command:

```bash
preprocess --dataset sleepedf --data_folder  "/your/data/path/"
```

To train the Chambon2018 model, run the following command ensuring the proper data path is set in the file:

```bash
python3 train.py
```

To train a simpler CNN model, run the following command:
```bash
python3 train_simple.py
```

To generate and evaluate the attributions with wavelets, run the following command for the simulated or sleepEDF:
```bash
python3 wavelets_mask.py
python3 wavelets_mask_sleepEDF.py
```

To generate and evaluate the attributions with FLEXtime masks, run the following command for the simulated or sleepEDF:
```bash
python3 flextime_simulated.py
python3 flextime_sleepedf.py
```


## Abstract

Over the past decade, Deep Learning (DL) models have been integrated into data-driven sectors such as e-commerce and healthcare to assist humans in making informed decisions. In neuroscience, these models have been utilized to analyze complex time series data, such as EEG and MEG, providing valuable insights for diagnosis and treatment. However, the black-box nature of this technology makes it challenging to provide the data foundation of those decisions and limits its trustworthiness. This research aims to propose a new explainability method that generates an attribution mask based on the multilevel discrete wavelet transform (DWT). Traditional approaches usually focus on the time or frequency domain, but this method simultaneously preserves and analyzes both. Even though the continuous wavelet transform (CWT) seems a better approach for signal processing and analysis, the DWT is more suitable for this task. It provides a set of orthogonal mother wavelets that allows a non-redundant and easily invertible transformation to frequency bands and time scales. The DWT has an advantage over the CWT, as it provides a more compact representation of the signal and allows for a more efficient implementation. Acting as a quadrature mirror filter bank, the DWT decomposes signals into approximation and detail coefficients across multiple levels, preserving critical information at various resolutions. To validate and evaluate the method, simulated datasets and the SleepEDF dataset were used. Both have been used for classification; the first one provides a defined ground truth, and the second establishes a well-defined classification problem, such as sleep staging. Additionally, various wavelet families, including Daubechies, Symlets, and Coiflets, have been tested and analyzed both qualitatively and quantitatively. The results show similar behavior between Symlets and Coiflets, due to their nearly symmetric nature. However, based on the Quantus complexity metric, Coiflets are less complex, which is preferred in the context of explainability. Moreover, the number of vanishing moments allows us to fine-tune the trade-off between frequency and time resolution, as well as computational cost. This research suggests that four vanishing moments offer a suitable balance between the various properties. Then, results show that while the learned attribution masks accurately highlight features in simulated scenarios, their performance on the real-world sleep staging task is limited. This is attributed to the DWT's coarse, dyadic frequency bands, which do not align well with the specific, non-dyadic frequency bands of sleep physiology. Therefore, this work introduces a promising framework, but highlights that its practical effectiveness depends on the suitability of the chosen transform for the specific problem domain.

## Contact

For further information, please contact Alicia García at aliciags99@gmail.com


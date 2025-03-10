#!/bin/bash
#BSUB -q gpuv100
#BSUB -J train                   # Job name
#BSUB -n 4                       # Number of cores
#BSUB -W 24:00                   # Wall-clock time (24 hours here)
#BSUB -R "rusage[mem=8GB]"       # Memory requirements (8 GB here)
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -o train_%J.out            # Standard output file
#BSUB -e train_%J.err            # Standard error file

# Activate the environment
source /zhome/2b/8/212341/learnable-masks-explainability-time-series/.env/bin/activate

# Run the Python script
python3 train.py

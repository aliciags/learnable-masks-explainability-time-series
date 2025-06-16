import pickle
import torch
import pandas as pd

with open("/work3/s241931/results/wavelets_results_5_sym10.pkl", "rb") as f:
    scores = pickle.load(f)

# Check the type of the loaded object
print(scores.keys())

def move_to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cpu()
    elif isinstance(obj, dict):
        return {k: move_to_cpu(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to_cpu(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to_cpu(v) for v in obj)
    else:
        return obj
    
scores = move_to_cpu(scores)

# store in another file
with open("/work3/s241931/results/wavelets_results_5_sym10_cpu.pkl", "wb") as f:
    pickle.dump(scores, f)
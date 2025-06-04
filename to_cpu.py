import pickle
import torch
import pandas as pd

with open("public/simple/_wavelets_results.pkl", "rb") as f:
    scores = pickle.load(f)

# Check the type of the loaded object
print(type(scores))  # likely <class 'list'> or <class 'tuple'>

# Check the length
print(len(scores))  # Output: 6
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

# # Now inspect each element
# for i, item in enumerate(scores.keys()):
#     t = type(scores[item])
#     print(f"Element {i}, key {item}: type={type(scores[item])}")

#     try:
#         if isinstance(scores[item], dict):
#             print(f"    keys={scores[item].keys()}")
#         elif isinstance(scores[item], list):
#             print(f"    length={len(scores[item])}")
#         else:
#             print(f"    shape={scores[item].shape}")
#             scores[item] = scores[item].to('cpu')
#     except AttributeError:
#         print("    No shape attribute")


# # df = pd.DataFrame(scores)
# # print(df.shape)
# # print(df.head())

# # set the dictionary to cpu??

# store in another file
with open("public/simple/_wavelets_cpu.pkl", "wb") as f:
    pickle.dump(scores, f)
import pickle
import pandas as pd

with open("public/sleepedf/flexime_128_501_32.pkl", "rb") as f:
    scores = pickle.load(f)

# Check the type of the loaded object
print(type(scores))  # likely <class 'list'> or <class 'tuple'>

# Check the length
print(len(scores))  # Output: 6
print(scores.keys())

# Now inspect each element
for i, item in enumerate(scores.keys()):
    t = type(scores[item])
    print(f"Element {i}, key {item}: type={type(scores[item])}")

    try:
        if isinstance(scores[item], dict):
            print(f"    keys={scores[item].keys()}")
        elif isinstance(scores[item], list):
            print(f"    length={len(scores[item])}")
        else:
            print(f"    shape={scores[item].shape}")
            scores[item] = scores[item].to('cpu')
    except AttributeError:
        print("    No shape attribute")


# df = pd.DataFrame(scores)
# print(df.shape)
# print(df.head())

# set the dictionary to cpu??

# store in another file
with open("public/sleepedf/flexime_128_501_32_cpu.pkl", "wb") as f:
    pickle.dump(scores, f)
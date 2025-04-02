import quantus
import numpy as np

# compute the complexity of the attribution
def compute_complexity_score(attribution):
    """
    Compute the complexity score of the attribution.
    """
    # compute the complexity score
    complexity = quantus.Complexity(return_aggregate=True)
    complexity_scores = []

    for i in range(len(attribution)):
        for j in range(len(attribution[i])):
            # compute the complexity score
            attr_sample = attribution[i][j].flatten().numpy()
            complexity_scores.append(np.nan_to_num(complexity.evaluate_instance(attr_sample, attr_sample)))

    return np.mean(complexity_scores)

def compute_batch_complexity_score(attribution):
    """
    Compute the complexity score of the attribution.
    """
    # compute the complexity score for the batch
    complexity = quantus.Complexity(return_aggregate=True)
    complexity_scores = []

    for i in range(len(attribution)):
        attr_batch = attribution[i].flatten(start_dim=1).numpy()
        complexity_scores += np.nan_to_num(complexity.evaluate_batch(attr_batch, attr_batch)).tolist()
    
    return np.mean(complexity_scores)

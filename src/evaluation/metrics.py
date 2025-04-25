import quantus
import numpy as np

# compute the complexity of the attribution
def compute_complexity_score_instance(data, attribution):
    """
    Compute the complexity score of the attribution per instance.

    Parameters
    ----------
    attribution : list
        The attribution scores.

    Returns
    -------
    float
        The mean complexity score.
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

def compute_complexity_score(attribution):
    """
    Compute the complexity score of the attribution per batch.

    Parameters
    ----------
    attribution : list
        The attribution scores.

    Returns
    -------
    float
        The mean complexity score.
    """
    # compute the complexity score for the batch
    complexity = quantus.Complexity()
    complexity_scores = []

    for i in range(len(attribution)):
        print(f"shape before flattening: {attribution[i].shape}")
        # attr_batch = attribution[i].flatten(start_dim=1).numpy()
        attr_batch = attribution[i]
        # print(f"shape after: {attr_batch.shape}")
        attr_batch = np.maximum(attr_batch, 0)
        complexity_scores += np.nan_to_num(complexity.evaluate_batch(attr_batch, attr_batch)).tolist()
    
    return np.mean(complexity_scores)

# compute the localization of the attribution
def compute_localization_score(data, attribution, masks):
    """
    Compute the localization score of the attribution.

    Parameters
    ----------
    data : list
        The data to be evaluated.
    attribution : list
        The attribution scores.
    mask : list
        The mask to be evaluated ?? - not sure might be ground truth??

    Returns
    -------
    float
        The mean localization score.
    """
    # compute the localization score
    localization = quantus.AttributionLocalisation()
    localization_scores = []

    # for each batch
    for i in range(len(attribution)):
        batch = data[i].numpy()
        attr = attribution[i].numpy()
        mask = masks[i]
        localization_scores.append(np.nan_to_num(localization.evaluate_batch(batch, attr, mask)))

    return np.mean(localization_scores)

# compute the faithfullness of the attribution
def compute_faithfullness_score(model, data, attributions, type: str = 'correlation'):
    """
    Compute the faithfullness score of the attribution.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be evaluated.
    data : list
        The data to be evaluated.
    attribution : list
        The attribution scores.

    Returns
    -------
    float
        The mean faithfullness score.
    """
    # compute the faithfullness score
    if type == 'estimate':
        faithfullness = quantus.FaithfulnessEstimate()
    else:
        faithfullness = quantus.FaithfulnessCorrelation()
    faithfullness_scores = []

    for i in range(len(attributions)):
        x, y = data[i]
        attr = attributions[i].numpy()
        faithfullness_scores.append(np.nan_to_num(faithfullness.evaluate_instance(model, x, y, attr)))

    return np.mean(faithfullness_scores)
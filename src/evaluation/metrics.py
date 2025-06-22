import quantus
import numpy as np
from sklearn.metrics import precision_recall_curve, auc

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
        attr_batch = attribution[i].squeeze(-1).detach().numpy().astype(np.float32)
        print(attr_batch)
        attr_batch = np.maximum(attr_batch, 0)
        print(attr_batch)
        complexities = complexity.evaluate_batch(attr_batch, attr_batch)
        print(complexities)
        complexity_scores += np.nan_to_num(complexities).tolist()

        if i == 3:
            break
    
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


def compute_wavelet_metrics(attributions, labels):
    """
    Compute AUR, AUP, and AUPRC metrics for wavelet attributions.
    
    Args:
        attributions: numpy array of shape (n_samples, time, n_filters)
        labels: numpy array of shape (n_samples, time)
    
    Returns:
        dict containing AUR, AUP, and AUPRC metrics
    """
    metrics = {}
    
    # Flatten the data
    n_samples = attributions.shape[0]
    n_filters = attributions.shape[2]
    
    # Initialize lists to store metrics
    aur_list = []
    aup_list = []
    auprc_list = []
    
    for i in range(n_samples):
        # Get the current sample's attributions and labels
        sample_attr = attributions[i]
        sample_labels = labels[i]
        
        # For each filter, compute the metrics
        for filter_idx in range(n_filters):
            # Get the current filter's attributions
            filter_attr = sample_attr[:, filter_idx]
            
            # Compute precision-recall curve
            precision, recall, _ = precision_recall_curve(sample_labels, filter_attr)
            
            # Compute AUR (area under recall curve)
            aur = auc(recall, recall)
            aur_list.append(aur)
            
            # Compute AUP (area under precision curve)
            aup = auc(precision, precision)
            aup_list.append(aup)
            
            # Compute AUPRC (area under precision-recall curve)
            auprc = auc(recall, precision)
            auprc_list.append(auprc)
    
    # Average metrics across all samples and filters
    metrics['AUR'] = np.mean(aur_list)
    metrics['AUP'] = np.mean(aup_list)
    metrics['AUPRC'] = np.mean(auprc_list)
    
    return metrics

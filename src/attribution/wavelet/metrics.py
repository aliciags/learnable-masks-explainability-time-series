import numpy as np
from sklearn.metrics import precision_recall_curve, auc

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

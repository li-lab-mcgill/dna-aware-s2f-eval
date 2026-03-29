import pdb
import os
import sys
import torch
import numpy as np
import scipy.stats as stats
import json
import torch.nn.functional as F
import pickle
import gzip
from copy import deepcopy


def fprint(*args, **kwargs):
    """Print with immediate flush"""
    print(*args, **kwargs, flush=True)

def compute_metrics(logits, targets, masks, ignore_index=-100):
    """
    Compute masked cross-entropy, accuracy, and self-entropy metrics
    
    Args:
        logits: (B, L, 4) model predictions
        targets: (B, L, 4) one-hot targets  
        masks: (B, L, 4) binary masks
        ignore_index: index to ignore in loss computation
    """
    # Convert targets to labels using preprocess_data
    labels = preprocess_data(targets, masks, ignore_index)  # (B, L)
    
    # Reshape for loss computation
    logits_flat = logits.view(-1, 4)       # (B*L, 4)
    labels_flat = labels.view(-1)          # (B*L,)
    
    # Get valid positions (not ignore_index)
    valid_mask = (labels_flat != ignore_index)
    total_positions = len(labels_flat)
    valid_positions = valid_mask.sum().item()
    valid_percentage = (valid_positions / total_positions) * 100
    
    fprint(f"  📊 TOKEN STATISTICS:")
    fprint(f"    Total positions: {total_positions:,}")
    fprint(f"    Valid positions: {valid_positions:,} ({valid_percentage:.2f}%)")
    fprint(f"    Invalid positions: {total_positions - valid_positions:,} ({100-valid_percentage:.2f}%)")
    
    if valid_mask.sum() == 0:
        fprint("🚨 ERROR: No valid positions found!")
        return {
            'cross_entropy': float('nan'),
            'accuracy': float('nan'),
            'matched_entropy': [],
            'unmatched_entropy': []
        }
    
    # Check for invalid logits
    if torch.isnan(logits_flat).any() or torch.isinf(logits_flat).any():
        nan_count = torch.isnan(logits_flat).sum().item()
        inf_count = torch.isinf(logits_flat).sum().item()
        fprint(f"🚨 ERROR: Invalid logits detected!")
        fprint(f"    NaN count: {nan_count:,}")
        fprint(f"    Inf count: {inf_count:,}")
        return {
            'cross_entropy': float('nan'),
            'accuracy': 0.0,
            'matched_entropy': [],
            'unmatched_entropy': []
        }
    
    # Get valid logits and labels
    valid_logits = logits_flat[valid_mask]      # (N_valid, 4)
    valid_labels = labels_flat[valid_mask]      # (N_valid,)
    
    fprint(f"  Valid logits range: [{valid_logits.min().item():.4f}, {valid_logits.max().item():.4f}]")
    fprint(f"  Valid labels unique: {torch.unique(valid_labels).tolist()}")
    
    # Compute cross-entropy with detailed error handling
    try:
        cross_entropy = F.cross_entropy(logits_flat, labels_flat, ignore_index=ignore_index)
        fprint(f"  ✅ Cross-entropy computed: {cross_entropy.item():.6f}")
        
        if torch.isnan(cross_entropy):
            fprint("🚨 ERROR: Cross-entropy is NaN!")
            fprint(f"    Valid samples: {valid_mask.sum()}")
            fprint(f"    Logits range: {valid_logits.min():.4f} to {valid_logits.max():.4f}")
            fprint(f"    Labels unique: {torch.unique(valid_labels)}")
            
            # Try manual computation to debug
            log_probs = F.log_softmax(valid_logits, dim=1)
            manual_ce = F.nll_loss(log_probs, valid_labels, reduction='mean')
            fprint(f"    Manual cross-entropy: {manual_ce.item():.6f}")
            
            cross_entropy = float('nan')
            
    except Exception as e:
        fprint(f"🚨 ERROR in cross_entropy calculation: {e}")
        fprint(f"    Exception type: {type(e).__name__}")
        cross_entropy = float('nan')
    
    # Compute predictions and accuracy
    predictions = valid_logits.argmax(dim=1)   # (N_valid,)
    accuracy = (predictions == valid_labels).float().mean()
    
    fprint(f"  ✅ Accuracy: {accuracy.item():.6f}")
    
    # Compute self-entropy for each valid position
    # Convert logits to probabilities
    probs = F.softmax(valid_logits, dim=1)     # (N_valid, 4)
    
    # Compute self-entropy: -sum(p * log(p)) for each position
    log_probs = F.log_softmax(valid_logits, dim=1)  # (N_valid, 4)
    entropies = -(probs * log_probs).sum(dim=1)     # (N_valid,) - entropy per position
    
    # Check entropy validity
    if torch.isnan(entropies).any():
        nan_entropy_count = torch.isnan(entropies).sum().item()
        fprint(f"🚨 WARNING: {nan_entropy_count} NaN entropies out of {len(entropies)}")
    
    # Determine which positions were correctly classified
    correct_mask = (predictions == valid_labels)    # (N_valid,)
    
    # Partition entropies based on correctness
    matched_entropies = entropies[correct_mask].tolist()      # List of entropies for correct predictions
    unmatched_entropies = entropies[~correct_mask].tolist()   # List of entropies for incorrect predictions
    
    fprint(f"  📈 ENTROPY STATS:")
    fprint(f"    Matched entropies: {len(matched_entropies):,} samples")
    fprint(f"    Unmatched entropies: {len(unmatched_entropies):,} samples")
    fprint(f"✅ COMPUTE_METRICS COMPLETE\n")
    
    return {
        'cross_entropy': cross_entropy.item() if not torch.isnan(cross_entropy) else float('nan'),
        'accuracy': accuracy.item(),
        'matched_entropy': matched_entropies,      # List of self-entropies for correctly classified positions
        'unmatched_entropy': unmatched_entropies   # List of self-entropies for incorrectly classified positions
    }

def preprocess_data(target, mask, ignore_index=-100):
    """
    Preprocess the labels for loss computation
    Args:
        target: torch.Tensor
            (B, L, 4) observed data
        mask: torch.Tensor
            (B, L, 4)  float / {0,1}
    Returns:
        dna_labels: torch.Tensor
            (B, L) DNA labels
    """
    # prepare DNA labels and mask 
    dna_labels = torch.argmax(target[:, :, :4], dim=2)  # (B, L)
    dna_mask = mask[..., :4].any(-1)  # (B, L)
    
    total_positions = dna_mask.numel()
    masked_positions = dna_mask.sum().item()
    mask_percentage = (masked_positions / total_positions) * 100
    
    fprint(f"  🎯 MASKING STATISTICS:")
    fprint(f"    Total positions: {total_positions:,}")
    fprint(f"    Masked positions: {masked_positions:,} ({mask_percentage:.2f}%)")
    fprint(f"    Unmasked positions: {total_positions - masked_positions:,} ({100-mask_percentage:.2f}%)")
    fprint(f"    DNA labels unique before masking: {torch.unique(dna_labels).tolist()}")
    
    # fill unmasked positions with ignore_index
    dna_labels = dna_labels.masked_fill(~dna_mask, ignore_index)
    
    final_valid = (dna_labels != ignore_index).sum().item()
    final_percentage = (final_valid / total_positions) * 100
    fprint(f"    Final valid labels: {final_valid:,} ({final_percentage:.2f}%)")
    fprint(f"    Final labels unique: {torch.unique(dna_labels).tolist()}")
    
    # return
    return dna_labels



def convert_to_json_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_json_serializable(i) for i in obj)
    else:
        return obj

def compute_entropy_statistics(entropy_list):
    """Compute comprehensive statistics for entropy distribution"""
    if len(entropy_list) == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "q1": 0.0,
            "median": 0.0,
            "q3": 0.0,
            "iqr": 0.0,
            "lower_whisker": 0.0,
            "upper_whisker": 0.0,
            "skewness": 0.0
        }
    
    entropies = np.array(entropy_list)
    
    # Basic statistics
    mean_val = float(np.mean(entropies))
    std_val = float(np.std(entropies))
    min_val = float(np.min(entropies))
    max_val = float(np.max(entropies))
    
    # Percentiles for box plot
    q1 = float(np.percentile(entropies, 25))
    median = float(np.percentile(entropies, 50))
    q3 = float(np.percentile(entropies, 75))
    iqr = q3 - q1
    
    # Whiskers (standard box plot definition)
    lower_whisker = max(min_val, q1 - 1.5 * iqr)
    upper_whisker = min(max_val, q3 + 1.5 * iqr)
    
    # Skewness
    skewness = float(stats.skew(entropies))
    
    return {
        "count": len(entropies),
        "mean": mean_val,
        "std": std_val,
        "min": min_val,
        "max": max_val,
        "q1": q1,
        "median": median,
        "q3": q3,
        "iqr": iqr,
        "lower_whisker": lower_whisker,
        "upper_whisker": upper_whisker,
        "skewness": skewness
    }

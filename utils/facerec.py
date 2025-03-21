import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy.spatial.distance import cosine
import pandas as pd

# Import our modules
from dataset.face_dataset import get_dataloaders
from models.pointnet import PointNetFaceRecognition
from train import ArcMarginProduct

def compute_embeddings(model, dataloader, device):
    """
    Compute embeddings for all samples in the dataloader
    
    Returns:
        embeddings: numpy array of embeddings
        labels: numpy array of labels
        ids: numpy array of sample ids (if available)
    """
    model.eval()
    
    all_embeddings = []
    all_labels = []
    all_ids = []
    
    with torch.no_grad():
        for data in tqdm(dataloader, desc='Computing embeddings'):
            points = data['points'].to(device)
            labels = data['label']
            
            # Get sample IDs if available
            ids = data.get('id', torch.zeros_like(labels))
            
            # Forward pass through PointNet to get embeddings
            _, _, _, embeddings = model(points)
            
            # Store embeddings, labels, and ids
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.numpy())
            all_ids.append(ids.numpy())
    
    return np.vstack(all_embeddings), np.hstack(all_labels), np.hstack(all_ids)

def plot_confusion_matrix(cm, class_names, output_path):
    """
    Plot and save confusion matrix
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_roc_curve(y_true, y_score, n_classes, output_path):
    """
    Plot and save ROC curves for multi-class classification
    """
    # Binarize the labels
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    
    plt.plot(fpr["micro"], tpr["micro"],
            label=f'micro-average ROC curve (area = {roc_auc["micro"]:.2f})',
            color='deeppink', linestyle=':', linewidth=4)
    
    # Only plot a subset of classes if there are many
    plot_classes = min(n_classes, 5)
    for i in range(plot_classes):
        plt.plot(fpr[i], tpr[i],
                label=f'ROC curve of class {i} (area = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC')
    plt.legend(loc="lower right")
    plt.savefig(output_path)
    plt.close()

def calculate_metrics(embeddings, labels, threshold=0.7):
    """
    Calculate verification metrics based on embeddings
    
    Args:
        embeddings: numpy array of embeddings
        labels: numpy array of labels (identity)
        threshold: cosine similarity threshold for positive match
    
    Returns:
        metrics: dict containing various verification metrics
    """
    # Normalize embeddings
    normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Initialize counters
    same_id_pairs = 0
    diff_id_pairs = 0
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    
    # Get all unique identities
    unique_ids = np.unique(labels)
    
    # Calculate a subset of pairs to evaluate (for efficiency)
    max_pairs = 10000  # Adjust based on dataset size
    pair_indices = []
    
    # Sample pairs with same identity
    for identity in unique_ids:
        idx = np.where(labels == identity)[0]
        if len(idx) >= 2:
            # Sample pairs with same identity
            same_id_samples = min(len(idx) * (len(idx) - 1) // 2, max_pairs // len(unique_ids))
            pairs = np.array([(i, j) for i in idx for j in idx if i < j])
            if len(pairs) > same_id_samples:
                pairs = pairs[np.random.choice(len(pairs), same_id_samples, replace=False)]
            pair_indices.extend([(i, j, 1) for i, j in pairs])  # 1 indicates same identity
    
    # Sample pairs with different identities
    diff_id_samples = min(len(pair_indices), max_pairs - len(pair_indices))
    diff_id_pairs_count = 0
    
    while diff_id_pairs_count < diff_id_samples:
        id1, id2 = np.random.choice(len(unique_ids), 2, replace=False)
        idx1 = np.where(labels == unique_ids[id1])[0]
        idx2 = np.where(labels == unique_ids[id2])[0]
        
        if len(idx1) > 0 and len(idx2) > 0:
            i = np.random.choice(idx1)
            j = np.random.choice(idx2)
            pair_indices.append((i, j, 0))  # 0 indicates different identity
            diff_id_pairs_count += 1
    
    # Evaluate all sampled pairs
    similarities = []
    pair_labels = []
    
    for i, j, is_same in pair_indices:
        # Calculate cosine similarity (higher means more similar)
        similarity = 1 - cosine(normalized_embeddings[i], normalized_embeddings[j])
        similarities.append(similarity)
        pair_labels.append(is_same)
        
        # Update counters
        if is_same == 1:
            same_id_pairs += 1
            if similarity >= threshold:
                true_positive += 1
            else:
                false_negative += 1
        else:
            diff_id_pairs += 1
            if similarity < threshold:
                true_negative += 1
            else:
                false_positive += 1
    
    # Calculate metrics
    accuracy = (true_positive + true_negative) / (same_id_pairs + diff_id_pairs) if (same_id_pairs + diff_id_pairs) > 0 else 0
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate FAR (False Accept Rate) and FRR (False Reject Rate)
    far = false_positive / diff_id_pairs if diff_id_pairs > 0 else 0
    frr = false_negative / same_id_pairs if same_id_pairs > 0 else 0
    
    # Return all metrics
    metrics = {
        'threshold': threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'far': far,
        'frr': frr,
        'same_id_pairs': same_id_pairs,
        'diff_id_pairs': diff_id_pairs,
        'true_positive': true_positive,
        'true_negative': true_negative,
        'false_positive': false_positive,
        'false_negative': false_negative
    }
    
    return metrics, np.array(similarities), np.array(pair_labels)

def find_best_threshold(similarities, pair_labels):
    """
    Find the best threshold that maximizes accuracy
    """
    best_acc = 0
    best_threshold = 0.5
    
    thresholds = np.linspace(0, 1, 100)
    
    for thresh in thresholds:
        predictions = (similarities >= thresh).astype(int)
        correct = np.sum(predictions == pair_labels)
        acc = correct / len(pair_labels)
        
        if acc > best_acc:
            best_acc = acc
            best_threshold = thresh
    
    return best_threshold, best_acc

def plot_verification_curves(similarities, pair_labels, output_dir):
    """
    Plot verification curves (ROC and precision-recall)
    """
    # Plot ROC curve
    fpr, tpr, thresholds = roc_curve(pair_labels, similarities)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, 'verification_roc.png'))
    plt.close()
    
    # Plot histogram of similarities
    plt.figure(figsize=(10, 8))
    sns.histplot(data=pd.DataFrame({
        'Similarity': similarities, 
        'Identity': ['Same' if l == 1 else 'Different' for l in pair_labels]
    }), x='Similarity', hue='Identity', bins=50, kde=True)
    plt.title('Distribution of Similarities')
    plt.savefig(os.path.join(output_dir, 'similarity_distribution.png'))
    plt.close()
    
    # Plot FAR-FRR curve
    thresholds = np.linspace(0, 1, 100)
    far_values = []
    frr_values = []
    
    for thresh in thresholds:
        # FAR: False positive rate for different identities
        far = np.sum((similarities >= thresh) & (pair_labels == 0)) / np.sum(pair_labels == 0)
        far_values.append(far)
        
        # FRR: False negative rate for same identities
        frr = np.sum((similarities < thresh) & (pair_labels == 1)) / np.sum(pair_labels == 1)
        frr_values.append(frr)
    
    # Find Equal Error Rate (EER)
    abs_diffs = np.abs(np.array(far_values) - np.array(frr_values))
    eer_idx = np.argmin(abs_diffs)
    eer = (far_values[eer_idx] + frr_values[eer_idx]) / 2
    eer_threshold = thresholds[eer_idx]
    
    plt.figure(figsize=(10, 8))
    plt.plot(thresholds, far_values, label='FAR (False Accept Rate)')
    plt.plot(thresholds, frr_values, label='FRR (False Reject Rate)')
    plt.axvline(x=eer_threshold, color='k', linestyle='--')
    plt.text(eer_threshold + 0.02, 0.5, f'EER: {eer:.4f} @ {eer_threshold:.4f}', 
             verticalalignment='center')
    plt.xlabel('Threshold')
    plt.ylabel('Error Rate')
    plt.title('FAR-FRR Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'far_frr_curve.png'))
    plt.close()
    
    return eer, eer_threshold

def evaluate_model(model_path, data_dir, output_dir, batch_size=32, num_points=1024, num_workers=4):
    """
    Evaluate a trained model
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get dataloaders
    _, test_loader, num_classes = get_dataloaders(
        data_dir,
        batch_size=batch_size,
        num_points=num_points,
        num_workers=num_workers
    )
    
    # Create model
    model = PointNetFaceRecognition(
        num_classes=num_classes,
        feature_transform=True,  # Assuming feature transform was used in training
        embedding_size=checkpoint['model_state_dict']['fc6.weight'].shape[1]  # Get embedding size from checkpoint
    ).to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create ArcFace metric if available in checkpoint
    if 'metric_fc_state_dict' in checkpoint:
        metric_fc = ArcMarginProduct(
            in_features=checkpoint['metric_fc_state_dict']['weight'].shape[1],
            out_features=checkpoint['metric_fc_state_dict']['weight'].shape[0]
        ).to(device)
        metric_fc.load_state_dict(checkpoint['metric_fc_state_dict'])
    else:
        metric_fc = None
    
    # Compute embeddings for test set
    print("Computing embeddings...")
    embeddings, labels, ids = compute_embeddings(model, test_loader, device)
    
    # Classification evaluation (if metric_fc is available)
    if metric_fc is not None:
        print("Evaluating classification performance...")
        # Convert embeddings to torch tensors
        embeddings_tensor = torch.tensor(embeddings).to(device)
        labels_tensor = torch.tensor(labels).to(device)
        
        # Forward pass through ArcFace
        with torch.no_grad():
            output = metric_fc(embeddings_tensor)
            probabilities = torch.softmax(output, dim=1).cpu().numpy()
            _, predictions = torch.max(output.data, 1)
            predictions = predictions.cpu().numpy()
        
        # Calculate metrics
        cm = confusion_matrix(labels, predictions)
        class_names = [str(i) for i in range(num_classes)]
        
        # Plot confusion matrix
        plot_confusion_matrix(cm, class_names, os.path.join(output_dir, 'confusion_matrix.png'))
        
        # Plot ROC curve
        plot_roc_curve(labels, probabilities, num_classes, os.path.join(output_dir, 'roc_curve.png'))
        
        # Classification report
        report = classification_report(labels, predictions, output_dict=True)
        with open(os.path.join(output_dir, 'classification_report.json'), 'w') as f:
            json.dump(report, f, indent=4)
        
        print(f"Classification accuracy: {report['accuracy']:.4f}")
    
    # Verification evaluation
    print("Evaluating verification performance...")
    
    # Calculate verification metrics with initial threshold
    metrics, similarities, pair_labels = calculate_metrics(embeddings, labels)
    
    # Find best threshold
    best_threshold, best_acc = find_best_threshold(similarities, pair_labels)
    
    # Recalculate metrics with best threshold
    metrics, _, _ = calculate_metrics(embeddings, labels, threshold=best_threshold)
    metrics['best_threshold'] = best_threshold
    
    # Plot verification curves
    eer, eer_threshold = plot_verification_curves(similarities, pair_labels, output_dir)
    metrics['eer'] = eer
    metrics['eer_threshold'] = eer_threshold
    
    # Save metrics
    with open(os.path.join(output_dir, 'verification_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Verification metrics:")
    print(f"  Best threshold: {best_threshold:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  FAR: {metrics['far']:.4f}")
    print(f"  FRR: {metrics['frr']:.4f}")
    print(f"  EER: {eer:.4f} @ threshold {eer_threshold:.4f}")
    
    return metrics

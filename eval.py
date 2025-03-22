import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from scipy.spatial.distance import cosine
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.units import inch
from datetime import datetime
from io import BytesIO

# Import our modules
from dataset.face_dataset import get_dataloaders
from models.pointnet import PointNetFaceRecognition
from models.pointnet2 import PointNet2FaceRecognition

def extract_embeddings(model, dataloader, device):
    """
    Extract embeddings for all samples in the dataloader
    
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
        for data in tqdm(dataloader, desc='Extracting embeddings'):
            points = data['points'].to(device)
            labels = data['label']
            
            # Get sample IDs if available
            ids = data.get('id', torch.zeros_like(labels))
            
            # Forward pass to get embeddings
            embeddings = model(points, get_embeddings=True)
            
            # Store embeddings, labels, and ids
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.numpy())
            all_ids.append(ids.numpy())
    
    return np.vstack(all_embeddings), np.hstack(all_labels), np.hstack(all_ids)

def calculate_similarity_metrics(embeddings, labels, threshold=0.7):
    """
    Calculate verification metrics based on embeddings
    
    Args:
        embeddings: numpy array of embeddings (already normalized)
        labels: numpy array of labels (identity)
        threshold: cosine similarity threshold for positive match
    
    Returns:
        metrics: dict containing various verification metrics
    """
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
        similarity = np.dot(embeddings[i], embeddings[j])  # Embeddings are already normalized
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

def find_optimal_threshold(similarities, pair_labels):
    """
    Find the optimal threshold for verification
    - Best accuracy threshold: maximizes accuracy
    - Equal Error Rate (EER) threshold: FAR = FRR
    """
    # Find best accuracy threshold
    best_acc = 0
    best_acc_threshold = 0.5
    
    thresholds = np.linspace(0, 1, 100)
    
    for thresh in thresholds:
        predictions = (similarities >= thresh).astype(int)
        correct = np.sum(predictions == pair_labels)
        acc = correct / len(pair_labels)
        
        if acc > best_acc:
            best_acc = acc
            best_acc_threshold = thresh
    
    # Find EER threshold
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
    
    return best_acc_threshold, best_acc, eer_threshold, eer, thresholds, far_values, frr_values

def create_verification_report(embeddings, labels, output_dir):
    """
    Create verification report with cosine similarity
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate verification metrics with initial threshold
    metrics, similarities, pair_labels = calculate_similarity_metrics(embeddings, labels, threshold=0.5)
    
    # Find optimal thresholds
    best_threshold, best_acc, eer_threshold, eer, thresholds, far_values, frr_values = find_optimal_threshold(similarities, pair_labels)
    
    # Recalculate metrics with best threshold
    metrics, _, _ = calculate_similarity_metrics(embeddings, labels, threshold=best_threshold)
    
    # Generate plots
    result_plots = {}
    
    # 1. ROC Curve
    plt.figure(figsize=(10, 8))
    fpr, tpr, _ = roc_curve(pair_labels, similarities)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    roc_path = os.path.join(output_dir, 'roc_curve.png')
    plt.savefig(roc_path)
    plt.close()
    result_plots['roc_curve'] = roc_path
    
    # 2. Similarity Distribution
    plt.figure(figsize=(10, 8))
    sns.histplot(data=pd.DataFrame({
        'Similarity': similarities, 
        'Identity': ['Same' if l == 1 else 'Different' for l in pair_labels]
    }), x='Similarity', hue='Identity', bins=50, kde=True)
    plt.title('Distribution of Similarities Between Face Pairs')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Count')
    
    dist_path = os.path.join(output_dir, 'similarity_distribution.png')
    plt.savefig(dist_path)
    plt.close()
    result_plots['similarity_distribution'] = dist_path
    
    # 3. FAR-FRR Curve
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
    
    far_frr_path = os.path.join(output_dir, 'far_frr_curve.png')
    plt.savefig(far_frr_path)
    plt.close()
    result_plots['far_frr_curve'] = far_frr_path
    
    # Create PDF report
    create_pdf_report(metrics, best_threshold, eer, eer_threshold, roc_auc, output_dir, result_plots)
    
    # Save metrics to JSON
    metrics.update({
        'best_threshold': best_threshold,
        'best_accuracy': best_acc,
        'eer': eer,
        'eer_threshold': eer_threshold,
        'roc_auc': roc_auc
    })
    
    with open(os.path.join(output_dir, 'verification_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Verification report created in {output_dir}")
    print(f"  Best threshold: {best_threshold:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  FAR: {metrics['far']:.4f}")
    print(f"  FRR: {metrics['frr']:.4f}")
    print(f"  EER: {eer:.4f} @ threshold {eer_threshold:.4f}")
    print(f"  ROC AUC: {roc_auc:.4f}")
    
    return metrics

def create_pdf_report(metrics, best_threshold, eer, eer_threshold, roc_auc, output_dir, plot_paths):
    """
    Create a PDF report with metrics and plots
    """
    pdf_path = os.path.join(output_dir, 'face_verification_report.pdf')
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    
    # Create styles
    styles = getSampleStyleSheet()
    title_style = styles['Title']
    heading_style = styles['Heading1']
    subheading_style = styles['Heading2']
    normal_style = styles['Normal']
    
    # Create content list
    content = []
    
    # Title
    content.append(Paragraph("3D Face Verification Report", title_style))
    content.append(Spacer(1, 0.25*inch))
    
    # Summary section
    content.append(Paragraph("1. Summary", heading_style))
    content.append(Spacer(1, 0.1*inch))
    
    # Metrics table
    content.append(Paragraph("Verification Metrics", subheading_style))
    
    summary_data = [
        ["Metric", "Value"],
        ["Best Accuracy Threshold", f"{best_threshold:.4f}"],
        ["Accuracy at Best Threshold", f"{metrics['accuracy']:.4f}"],
        ["Precision", f"{metrics['precision']:.4f}"],
        ["Recall", f"{metrics['recall']:.4f}"],
        ["F1 Score", f"{metrics['f1']:.4f}"],
        ["False Accept Rate (FAR)", f"{metrics['far']:.4f}"],
        ["False Reject Rate (FRR)", f"{metrics['frr']:.4f}"],
        ["Equal Error Rate (EER)", f"{eer:.4f}"],
        ["EER Threshold", f"{eer_threshold:.4f}"],
        ["ROC AUC", f"{roc_auc:.4f}"],
        ["Total Same-Identity Pairs", f"{metrics['same_id_pairs']}"],
        ["Total Different-Identity Pairs", f"{metrics['diff_id_pairs']}"]
    ]
    
    metrics_table = Table(summary_data, colWidths=[3*inch, 2.5*inch])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    content.append(metrics_table)
    content.append(Spacer(1, 0.2*inch))
    
    # ROC Curve
    content.append(Paragraph("2. ROC Curve", heading_style))
    content.append(Spacer(1, 0.1*inch))
    
    content.append(Paragraph("The Receiver Operating Characteristic (ROC) curve shows the trade-off between true positive rate and false positive rate at different thresholds.", normal_style))
    content.append(Spacer(1, 0.1*inch))
    
    img = Image(plot_paths['roc_curve'], width=6*inch, height=5*inch)
    content.append(img)
    content.append(Spacer(1, 0.2*inch))
    
    # Similarity Distribution
    content.append(Paragraph("3. Similarity Distribution", heading_style))
    content.append(Spacer(1, 0.1*inch))
    
    content.append(Paragraph("The distribution of cosine similarities between same-identity and different-identity face pairs:", normal_style))
    content.append(Spacer(1, 0.1*inch))
    
    img = Image(plot_paths['similarity_distribution'], width=6*inch, height=5*inch)
    content.append(img)
    content.append(Spacer(1, 0.2*inch))
    
    # FAR-FRR Curve
    content.append(Paragraph("4. FAR-FRR Curve", heading_style))
    content.append(Spacer(1, 0.1*inch))
    
    content.append(Paragraph("The False Accept Rate (FAR) and False Reject Rate (FRR) at different thresholds. The Equal Error Rate (EER) is the point where FAR equals FRR.", normal_style))
    content.append(Spacer(1, 0.1*inch))
    
    img = Image(plot_paths['far_frr_curve'], width=6*inch, height=5*inch)
    content.append(img)
    content.append(Spacer(1, 0.2*inch))
    
    # Conclusions
    content.append(Paragraph("5. Conclusions and Recommendations", heading_style))
    content.append(Spacer(1, 0.1*inch))
    
    # Generate assessment based on metrics
    if eer < 0.05:
        eer_assessment = "excellent"
    elif eer < 0.1:
        eer_assessment = "good"
    elif eer < 0.15:
        eer_assessment = "acceptable"
    else:
        eer_assessment = "needs improvement"
    
    conclusion_text = f"""
    The 3D face verification system achieves an Equal Error Rate (EER) of {eer:.4f}, which is {eer_assessment}. 
    
    For optimal accuracy, a similarity threshold of {best_threshold:.4f} is recommended, which yields:
    - Accuracy: {metrics['accuracy']:.4f}
    - Precision: {metrics['precision']:.4f}
    - Recall: {metrics['recall']:.4f}
    
    The ROC AUC of {roc_auc:.4f} indicates {'excellent' if roc_auc > 0.95 else 'good' if roc_auc > 0.9 else 'acceptable' if roc_auc > 0.8 else 'poor'} discriminative power.
    """
    content.append(Paragraph(conclusion_text, normal_style))
    content.append(Spacer(1, 0.1*inch))
    
    # Add recommendations
    recommendations = """
    Recommendations for improvement:
    
    1. If the EER is above 0.1, consider:
       - Collecting more training data
       - Improving the quality of 3D face scans
       - Increasing the embedding dimensionality
    
    2. If the similarity distributions have significant overlap:
       - Try increasing the margin in the loss function
       - Explore different feature extraction architectures
    
    3. For production deployment:
       - Choose a threshold based on the specific security requirements
       - For high security, use a higher threshold (lower FAR, higher FRR)
       - For convenience, use a lower threshold (higher FAR, lower FRR)
    """
    content.append(Paragraph(recommendations, normal_style))
    
    # Build the PDF
    doc.build(content)
    print(f"PDF report generated at: {pdf_path}")

def evaluate_embeddings(model_path, data_dir, output_dir, batch_size=32, num_points=1024, num_workers=4):
    """
    Evaluate a trained model using embeddings and cosine similarity
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    print(f"Using device: {device}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get dataloaders
    test_loader = get_dataloaders(
        data_dir,
        batch_size=batch_size,
        num_points=num_points,
        num_workers=num_workers,
        only_one=True
    )
    
    # Create model
    embedding_size = 512
    print(f"Using embedding size: {embedding_size}")
    
    model = PointNet2FaceRecognition(
        num_classes=123,
        feature_transform=False,  # Assuming feature transform was used in training
        embedding_size=embedding_size
    ).to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from {model_path}")
    
    # Extract embeddings
    print("Extracting embeddings...")
    embeddings, labels, ids = extract_embeddings(model, test_loader, device)
    print(f"Extracted {len(embeddings)} embeddings with shape {embeddings.shape}")
    
    # Create verification report
    metrics = create_verification_report(embeddings, labels, output_dir)
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='3D Face Recognition Embedding Evaluation')
    
    # Data parameters
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True, help='Data directory')
    parser.add_argument('--output_dir', type=str, default='embedding_evaluation', help='Output directory for evaluation results')
    
    # Evaluation parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_points', type=int, default=4096, help='Number of points to sample')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers')
    parser.add_argument('--device', type=str, default="cpu", help='Number of dataloader workers')

    args = parser.parse_args()
    
    # Evaluate model using embeddings
    evaluate_embeddings(
        model_path=args.model_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_points=args.num_points,
        num_workers=args.num_workers
    )

if __name__ == '__main__':
    main()

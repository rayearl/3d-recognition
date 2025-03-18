
# Note: this is untested and direct from Claude.AI

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
import open3d as o3d
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import our model architecture
from model import FacePointCloudNet, TripletLoss

###########################################
# Point Cloud Data Preprocessing Functions
###########################################

def normalize_point_cloud(points):
    """
    Normalize point cloud by centering and scaling to unit sphere
    
    Args:
        points: numpy array of shape (N, 3) for XYZ coordinates
        
    Returns:
        normalized points
    """
    # Find center of mass
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid
    
    # Scale to unit sphere
    max_distance = np.max(np.sqrt(np.sum(points_centered**2, axis=1)))
    points_normalized = points_centered / max_distance
    
    return points_normalized

def normalize_rgb_values(rgb_values):
    """
    Normalize RGB values to [0, 1] range
    
    Args:
        rgb_values: numpy array of shape (N, 3) for RGB values
        
    Returns:
        normalized RGB values
    """
    return rgb_values / 255.0

def sample_point_cloud(points, rgb_values, num_points):
    """
    Sample a fixed number of points from the point cloud
    
    Args:
        points: numpy array of shape (N, 3) for XYZ coordinates
        rgb_values: numpy array of shape (N, 3) for RGB values
        num_points: number of points to sample
        
    Returns:
        sampled points and RGB values
    """
    # If we have fewer points than requested, duplicate points
    if points.shape[0] < num_points:
        # Calculate how many times to duplicate and how many remaining points
        repeat_factor = num_points // points.shape[0]
        remainder = num_points % points.shape[0]
        
        # Duplicate points
        if repeat_factor > 0:
            points_repeated = np.repeat(points, repeat_factor, axis=0)
            rgb_repeated = np.repeat(rgb_values, repeat_factor, axis=0)
            
            # Add remaining points
            if remainder > 0:
                indices = np.random.choice(points.shape[0], remainder, replace=False)
                points_sampled = np.concatenate([points_repeated, points[indices]], axis=0)
                rgb_sampled = np.concatenate([rgb_repeated, rgb_values[indices]], axis=0)
            else:
                points_sampled = points_repeated
                rgb_sampled = rgb_repeated
        else:
            # Just sample the required number
            indices = np.random.choice(points.shape[0], num_points, replace=False)
            points_sampled = points[indices]
            rgb_sampled = rgb_values[indices]
    else:
        # Randomly sample points
        indices = np.random.choice(points.shape[0], num_points, replace=False)
        points_sampled = points[indices]
        rgb_sampled = rgb_values[indices]
    
    return points_sampled, rgb_sampled

def rotate_point_cloud(points, max_angle=np.pi/18):
    """
    Randomly rotate the point cloud around Z axis (assuming face is frontal)
    
    Args:
        points: numpy array of shape (N, 3) for XYZ coordinates
        max_angle: maximum rotation angle in radians
        
    Returns:
        rotated points
    """
    angle = np.random.uniform(-max_angle, max_angle)
    cosval = np.cos(angle)
    sinval = np.sin(angle)
    rotation_matrix = np.array([[cosval, -sinval, 0],
                                [sinval, cosval, 0],
                                [0, 0, 1]])
    return np.dot(points, rotation_matrix)

def jitter_points(points, sigma=0.01, clip=0.05):
    """
    Randomly jitter points by adding Gaussian noise
    
    Args:
        points: numpy array of shape (N, 3) for XYZ coordinates
        sigma: standard deviation of Gaussian noise
        clip: maximum noise magnitude
        
    Returns:
        jittered points
    """
    noise = np.clip(sigma * np.random.randn(*points.shape), -clip, clip)
    return points + noise

def jitter_colors(rgb_values, sigma=0.01, clip=0.05):
    """
    Randomly jitter colors by adding Gaussian noise
    
    Args:
        rgb_values: numpy array of shape (N, 3) for RGB values (assumed to be in [0, 1])
        sigma: standard deviation of Gaussian noise
        clip: maximum noise magnitude
        
    Returns:
        jittered RGB values, clipped to [0, 1]
    """
    noise = np.clip(sigma * np.random.randn(*rgb_values.shape), -clip, clip)
    jittered = rgb_values + noise
    return np.clip(jittered, 0, 1)

def load_point_cloud_from_file(file_path):
    """
    Load point cloud from file (supports various formats via Open3D)
    
    Args:
        file_path: path to the point cloud file
        
    Returns:
        xyz points and rgb values as numpy arrays
    """
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)  # Open3D uses [0, 1] range for colors
    
    # Convert mm to normalized coordinates
    points_normalized = normalize_point_cloud(points)
    
    return points_normalized, colors

def augment_point_cloud(points, rgb_values, augment=True):
    """
    Apply augmentations to point cloud data
    
    Args:
        points: numpy array of shape (N, 3) for XYZ coordinates
        rgb_values: numpy array of shape (N, 3) for RGB values
        augment: whether to apply augmentation
        
    Returns:
        augmented points and RGB values
    """
    if not augment:
        return points, rgb_values
    
    # Apply random rotation around z-axis (assuming face is frontal)
    points = rotate_point_cloud(points)
    
    # Apply random jittering to points
    points = jitter_points(points)
    
    # Apply random jittering to colors
    rgb_values = jitter_colors(rgb_values)
    
    return points, rgb_values

#################################
# Dataset and DataLoader Classes
#################################

class FacePointCloudDataset(Dataset):
    """
    Dataset for face point clouds
    
    Args:
        file_list: list of file paths for point clouds
        labels: corresponding identity labels
        num_points: number of points to sample from each point cloud
        transform: whether to apply data augmentation
    """
    def __init__(self, file_list, labels, num_points=1024, transform=False):
        self.file_list = file_list
        self.labels = labels
        self.num_points = num_points
        self.transform = transform
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        # Load point cloud
        points, rgb_values = load_point_cloud_from_file(self.file_list[idx])
        
        # Sample fixed number of points
        points, rgb_values = sample_point_cloud(points, rgb_values, self.num_points)
        
        # Apply augmentation if enabled
        if self.transform:
            points, rgb_values = augment_point_cloud(points, rgb_values)
        
        # Combine points and colors into a single tensor
        # Shape: [N, 6] where first 3 are XYZ, next 3 are RGB
        point_data = np.concatenate([points, rgb_values], axis=1)
        
        # Convert to PyTorch tensor
        point_data_tensor = torch.from_numpy(point_data).float()
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return point_data_tensor, label_tensor

def prepare_data_loaders(data_dir, batch_size=32, num_points=1024, num_samples_per_identity=4):
    """
    Prepare data loaders for training and validation
    
    Args:
        data_dir: root directory containing point cloud files
        batch_size: batch size for training
        num_points: number of points to sample from each point cloud
        num_samples_per_identity: minimum number of samples per identity for triplet formation
        
    Returns:
        train_loader, val_loader
    """
    # Collect all file paths and labels
    file_paths = []
    labels = []
    identity_to_label = {}
    label_counter = 0
    
    print(f"Loading data from {data_dir}...")
    
    # Assuming directory structure: data_dir/identity/scan_files
    # This ensures multiple scans per identity for triplet formation
    for identity in os.listdir(data_dir):
        identity_dir = os.path.join(data_dir, identity)
        if os.path.isdir(identity_dir):
            identity_files = []
            for file_name in os.listdir(identity_dir):
                if file_name.endswith(('.ply', '.pcd', '.xyz')):  # Common point cloud formats
                    file_path = os.path.join(identity_dir, file_name)
                    identity_files.append(file_path)
            
            # Only include identities with enough samples
            if len(identity_files) >= num_samples_per_identity:
                identity_to_label[identity] = label_counter
                for file_path in identity_files:
                    file_paths.append(file_path)
                    labels.append(label_counter)
                label_counter += 1
    
    print(f"Found {len(file_paths)} point clouds across {label_counter} identities")
    
    # Split into training and validation sets
    train_files, val_files, train_labels, val_labels = train_test_split(
        file_paths, labels, test_size=0.2, stratify=labels, random_state=42
    )
    
    # Create datasets
    train_dataset = FacePointCloudDataset(
        train_files, train_labels, num_points=num_points, transform=True
    )
    val_dataset = FacePointCloudDataset(
        val_files, val_labels, num_points=num_points, transform=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    
    return train_loader, val_loader

#################################
# Training Functions
#################################

def train_epoch(model, train_loader, optimizer, criterion, device):
    """
    Train for one epoch
    
    Args:
        model: the face point cloud model
        train_loader: data loader for training data
        optimizer: optimizer for training
        criterion: triplet loss
        device: device to train on
        
    Returns:
        average loss for the epoch
    """
    model.train()
    total_loss = 0
    
    # Progress bar
    pbar = tqdm(train_loader, desc="Training")
    
    for batch_idx, (data, labels) in enumerate(pbar):
        # Move data to device
        data, labels = data.to(device), labels.to(device)
        
        # Reset gradients
        optimizer.zero_grad()
        
        # Forward pass
        embeddings = model(data)
        
        # Calculate loss
        loss = criterion(embeddings, labels)
        
        # Backpropagate
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Update statistics
        total_loss += loss.item()
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item()})
    
    # Return average loss
    return total_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    """
    Validate the model
    
    Args:
        model: the face point cloud model
        val_loader: data loader for validation data
        criterion: triplet loss
        device: device to validate on
        
    Returns:
        average loss for validation set
    """
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for data, labels in tqdm(val_loader, desc="Validation"):
            # Move data to device
            data, labels = data.to(device), labels.to(device)
            
            # Forward pass
            embeddings = model(data)
            
            # Calculate loss
            loss = criterion(embeddings, labels)
            
            # Update statistics
            total_loss += loss.item()
    
    # Return average loss
    return total_loss / len(val_loader)

def compute_accuracy(embeddings, labels, threshold=0.7):
    """
    Compute face verification accuracy
    
    Args:
        embeddings: face embeddings
        labels: identity labels
        threshold: cosine similarity threshold for verification
        
    Returns:
        accuracy, true_positive_rate, false_positive_rate
    """
    # Normalize embeddings
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    
    # Compute pairwise cosine similarity
    similarity_matrix = torch.matmul(embeddings, embeddings.transpose(0, 1))
    
    # Create ground truth matrix (1 if same identity, 0 otherwise)
    label_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)
    
    # Only consider off-diagonal elements (ignore self-comparisons)
    mask = ~torch.eye(similarity_matrix.shape[0], dtype=torch.bool, device=similarity_matrix.device)
    similarity_matrix = similarity_matrix[mask].flatten()
    label_matrix = label_matrix[mask].flatten()
    
    # Create predicted matrix based on threshold
    pred_matrix = (similarity_matrix > threshold).float()
    
    # Compute metrics
    correct = (pred_matrix == label_matrix.float()).sum().item()
    total = label_matrix.size(0)
    
    # True positive rate (sensitivity)
    tp = (pred_matrix * label_matrix.float()).sum().item()
    actual_positives = label_matrix.sum().item()
    tpr = tp / actual_positives if actual_positives > 0 else 0
    
    # False positive rate (1 - specificity)
    fp = (pred_matrix * (1 - label_matrix.float())).sum().item()
    actual_negatives = (1 - label_matrix.float()).sum().item()
    fpr = fp / actual_negatives if actual_negatives > 0 else 0
    
    return correct / total, tpr, fpr

def compute_verification_metrics(model, val_loader, device):
    """
    Compute verification metrics on validation set
    
    Args:
        model: the face point cloud model
        val_loader: data loader for validation data
        device: device to validate on
        
    Returns:
        dictionary with metrics
    """
    model.eval()
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in tqdm(val_loader, desc="Computing metrics"):
            # Move data to device
            data = data.to(device)
            
            # Forward pass
            embeddings = model(data)
            
            # Collect embeddings and labels
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels)
    
    # Concatenate all embeddings and labels
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Move to device for computation
    all_embeddings = all_embeddings.to(device)
    all_labels = all_labels.to(device)
    
    # Compute metrics for different thresholds
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    results = {}
    
    for threshold in thresholds:
        accuracy, tpr, fpr = compute_accuracy(all_embeddings, all_labels, threshold)
        results[f"accuracy_{threshold:.1f}"] = accuracy
        results[f"tpr_{threshold:.1f}"] = tpr
        results[f"fpr_{threshold:.1f}"] = fpr
    
    return results

def plot_training_progress(train_losses, val_losses, metrics_history, save_path="training_progress.png"):
    """
    Plot training progress
    
    Args:
        train_losses: list of training losses
        val_losses: list of validation losses
        metrics_history: dictionary of metrics history
        save_path: path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b', label='Training loss')
    ax1.plot(epochs, val_losses, 'r', label='Validation loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Plot metrics
    if metrics_history:
        for key in metrics_history:
            if key.startswith('accuracy'):
                threshold = key.split('_')[-1]
                ax2.plot(epochs, metrics_history[key], label=f'Accuracy (t={threshold})')
        
        ax2.set_title('Verification Accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
    
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()

def train_model(data_dir, model_save_dir, num_epochs=50, batch_size=32, learning_rate=0.001, 
                embedding_size=512, num_points=1024, margin=0.2, device_str="cuda"):
    """
    Train the face point cloud model
    
    Args:
        data_dir: root directory containing point cloud files
        model_save_dir: directory to save model checkpoints
        num_epochs: number of epochs to train
        batch_size: batch size for training
        learning_rate: initial learning rate
        embedding_size: size of face embedding
        num_points: number of points to sample from each point cloud
        margin: margin for triplet loss
        device_str: device to train on
        
    Returns:
        trained model
    """
    # Create save directory if it doesn't exist
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Set device
    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    print(f"Using device: {device}")
    
    # Prepare data loaders
    train_loader, val_loader = prepare_data_loaders(
        data_dir, batch_size=batch_size, num_points=num_points
    )
    
    # Create model
    model = FacePointCloudNet(embedding_size=embedding_size, num_points=num_points)
    model = model.to(device)
    
    # Create criterion
    criterion = TripletLoss(margin=margin)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
    
    # Training history
    train_losses = []
    val_losses = []
    metrics_history = {}
    best_val_loss = float('inf')
    
    print(f"Starting training for {num_epochs} epochs")
    
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        
        # Train for one epoch
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        # Update learning rate
        scheduler.step()
        
        # Compute verification metrics every 5 epochs
        if epoch % 5 == 0 or epoch == num_epochs:
            metrics = compute_verification_metrics(model, val_loader, device)
            
            # Initialize metric history if first time
            if not metrics_history:
                for key in metrics:
                    metrics_history[key] = []
            
            # Update metric history
            for key, value in metrics.items():
                metrics_history[key].append(value)
            
            # Print metrics
            print("Verification Metrics:")
            for key, value in metrics.items():
                print(f"  {key}: {value:.4f}")
        
        # Print epoch summary
        print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        
        # Save checkpoint if best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(model_save_dir, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'metrics': metrics if epoch % 5 == 0 or epoch == num_epochs else None
            }, checkpoint_path)
            print(f"Model saved to {checkpoint_path}")
        
        # Save latest model
        latest_path = os.path.join(model_save_dir, "latest_model.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'metrics': metrics if epoch % 5 == 0 or epoch == num_epochs else None
        }, latest_path)
        
        # Plot training progress
        plot_training_progress(train_losses, val_losses, metrics_history, 
                              save_path=os.path.join(model_save_dir, "training_progress.png"))
    
    print("Training complete!")
    return model

#################################
# Example Usage
#################################

if __name__ == "__main__":
    # Example configuration
    config = {
        "data_dir": "/Volumes/T7 Shield/ARC Images/3D synthetic - FPC/sample_output",
        "model_save_dir": "./saved_models",
        "num_epochs": 25,
        "batch_size": 16,
        "learning_rate": 0.001,
        "embedding_size": 512,
        "num_points": 1024,
        "margin": 0.2,
        "device": "metal"
    }
    
    # Train model
    model = train_model(
        data_dir=config["data_dir"],
        model_save_dir=config["model_save_dir"],
        num_epochs=config["num_epochs"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        embedding_size=config["embedding_size"],
        num_points=config["num_points"],
        margin=config["margin"],
        device_str=config["device"]
    )
    
    # Example of how to load the model for inference
    def load_model_for_inference(model_path, embedding_size=512, num_points=1024, device_str="cuda"):
        device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
        model = FacePointCloudNet(embedding_size=embedding_size, num_points=num_points)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        return model
    
    # Example of face verification
    def verify_faces(model, face1_path, face2_path, threshold=0.7, num_points=1024, device_str="cuda"):
        device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
        
        # Load and preprocess faces
        points1, rgb1 = load_point_cloud_from_file(face1_path)
        points2, rgb2 = load_point_cloud_from_file(face2_path)
        
        # Sample points
        points1, rgb1 = sample_point_cloud(points1, rgb1, num_points)
        points2, rgb2 = sample_point_cloud(points2, rgb2, num_points)
        
        # Combine points and colors
        point_data1 = np.concatenate([points1, rgb1], axis=1)
        point_data2 = np.concatenate([points2, rgb2], axis=1)
        
        # Convert to tensors
        point_data1 = torch.from_numpy(point_data1).float().unsqueeze(0).to(device)
        point_data2 = torch.from_numpy(point_data2).float().unsqueeze(0).to(device)
        
        # Get embeddings
        with torch.no_grad():
            embedding1 = model(point_data1)
            embedding2 = model(point_data2)
        
        # Normalize embeddings
        embedding1 = torch.nn.functional.normalize(embedding1, p=2, dim=1)
        embedding2 = torch.nn.functional.normalize(embedding2, p=2, dim=1)
        
        # Compute cosine similarity
        similarity = torch.matmul(embedding1, embedding2.transpose(0, 1)).item()
        
        # Verify
        is_same_person = similarity > threshold
        
        return {
            "is_same_person": is_same_person,
            "similarity_score": similarity,
            "threshold": threshold
        }
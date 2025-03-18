import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.model_selection import train_test_split
import trimesh
from tqdm import tqdm
import matplotlib.pyplot as plt

# Set memory management environment variables
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

# Import our optimized model architecture
from model_3d_claude import FacePointCloudNetLite, TripletLossMemoryEfficient

# Point Cloud Data Preprocessing Functions
def normalize_point_cloud(points):
    """
    Normalize point cloud by centering and scaling to unit sphere
    """
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid
    max_distance = np.max(np.sqrt(np.sum(points_centered**2, axis=1)))
    points_normalized = points_centered / max_distance
    return points_normalized

def normalize_rgb_values(rgb_values):
    """
    Normalize RGB values to [0, 1] range
    """
    return rgb_values / 255.0

def load_point_cloud_from_file(file_path):
    """
    Load point cloud from file using trimesh (more macOS compatible than Open3D)
    """
    # Load the mesh or point cloud
    mesh = trimesh.load(file_path)
    
    # Extract points
    if hasattr(mesh, 'vertices'):
        # For mesh files
        points = np.array(mesh.vertices)
        # Extract colors if available
        if hasattr(mesh.visual, 'vertex_colors'):
            # Trimesh stores colors as RGBA uint8 (0-255)
            colors = mesh.visual.vertex_colors[:, :3] / 255.0
        else:
            # Generate random colors if no color info available
            colors = np.ones((points.shape[0], 3)) * 0.5  # Default gray color
    else:
        # For point cloud files
        points = np.array(mesh.vertices)
        
        # Extract colors if available
        if hasattr(mesh, 'colors'):
            colors = np.array(mesh.colors[:, :3]) / 255.0
        elif hasattr(mesh.visual, 'vertex_colors'):
            colors = mesh.visual.vertex_colors[:, :3] / 255.0
        else:
            # Generate random colors if no color info available
            colors = np.ones((points.shape[0], 3)) * 0.5  # Default gray color
    
    # Convert mm to normalized coordinates
    points_normalized = normalize_point_cloud(points)
    
    return points_normalized, colors

def sample_point_cloud(points, rgb_values, num_points):
    """
    Sample a fixed number of points from the point cloud - memory-efficient version
    """
    # If we have more points than needed, use deterministic sampling
    if points.shape[0] > num_points:
        # Use deterministic sampling for stability
        indices = np.linspace(0, points.shape[0]-1, num_points, dtype=int)
        points_sampled = points[indices]
        rgb_sampled = rgb_values[indices]
    else:
        # If we have fewer points than needed, duplicate and/or sample
        repeat_factor = num_points // points.shape[0]
        remainder = num_points % points.shape[0]
        
        if repeat_factor > 0:
            # Use tile instead of repeat for memory efficiency
            points_repeated = np.tile(points, (repeat_factor, 1))
            rgb_repeated = np.tile(rgb_values, (repeat_factor, 1))
            
            if remainder > 0:
                # Use deterministic sampling for the remainder
                indices = np.linspace(0, points.shape[0]-1, remainder, dtype=int)
                points_sampled = np.vstack([points_repeated, points[indices]])
                rgb_sampled = np.vstack([rgb_repeated, rgb_values[indices]])
            else:
                points_sampled = points_repeated
                rgb_sampled = rgb_repeated
        else:
            # Just sample the required number deterministically
            indices = np.linspace(0, points.shape[0]-1, num_points, dtype=int)
            points_sampled = points[indices]
            rgb_sampled = rgb_values[indices]
    
    return points_sampled, rgb_sampled

def augment_point_cloud(points, rgb_values, augment=True):
    """
    Apply light augmentations to point cloud data
    """
    if not augment:
        return points, rgb_values
    
    # Reduced augmentation intensity to save memory
    # Apply small random rotation around z-axis
    angle = np.random.uniform(-np.pi/36, np.pi/36)  # ±5 degrees
    cosval = np.cos(angle)
    sinval = np.sin(angle)
    rotation_matrix = np.array([[cosval, -sinval, 0],
                                [sinval, cosval, 0],
                                [0, 0, 1]])
    points = np.dot(points, rotation_matrix)
    
    # Apply very light jittering to points
    sigma = 0.005  # Reduced from 0.01
    clip = 0.02    # Reduced from 0.05
    noise = np.clip(sigma * np.random.randn(*points.shape), -clip, clip)
    points = points + noise
    
    # Apply light jittering to colors
    sigma = 0.005  # Reduced from 0.01
    clip = 0.02    # Reduced from 0.05
    noise = np.clip(sigma * np.random.randn(*rgb_values.shape), -clip, clip)
    rgb_values = np.clip(rgb_values + noise, 0, 1)
    
    return points, rgb_values

class FacePointCloudDataset(Dataset):
    """
    Dataset for face point clouds
    """
    def __init__(self, file_list, labels, num_points=512, transform=False):
        self.file_list = file_list
        self.labels = labels
        self.num_points = num_points
        self.transform = transform
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        
        # Check if this is a numpy file (preprocessed)
        if file_path.endswith('_points.npy'):
            # Load directly from numpy files
            points = np.load(file_path)
            colors_path = file_path.replace('_points.npy', '_colors.npy')
            if os.path.exists(colors_path):
                rgb_values = np.load(colors_path)
            else:
                # Generate default colors
                rgb_values = np.ones((points.shape[0], 3)) * 0.5
        else:
            # Load from 3D file format
            points, rgb_values = load_point_cloud_from_file(file_path)
        
        # Sample fixed number of points
        points, rgb_values = sample_point_cloud(points, rgb_values, self.num_points)
        
        # Apply augmentation if enabled (lighter augmentation)
        if self.transform:
            points, rgb_values = augment_point_cloud(points, rgb_values)
        
        # Combine points and colors into a single tensor
        # Shape: [N, 6] where first 3 are XYZ, next 3 are RGB
        point_data = np.concatenate([points, rgb_values], axis=1)
        
        # Convert to PyTorch tensor - use float32 to save memory
        point_data_tensor = torch.from_numpy(point_data).float()
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return point_data_tensor, label_tensor

def collect_files_and_labels(directory, min_samples_per_identity):
    """
    Collect files and their corresponding labels from a directory
    """
    file_paths = []
    labels = []
    identity_to_label = {}
    label_counter = 0
    
    # Check if this is a preprocessed numpy dataset
    preprocessed = False
    for root, dirs, files in os.walk(directory):
        if any(f.endswith("_points.npy") for f in files):
            preprocessed = True
            break
    
    # Process each identity directory
    for identity in os.listdir(directory):
        # Skip hidden directories (those starting with ".")
        if identity.startswith('.'):
            continue
            
        identity_dir = os.path.join(directory, identity)
        if os.path.isdir(identity_dir):
            identity_files = []
            
            # Check files in this identity directory
            for file_name in os.listdir(identity_dir):
                # Skip hidden files and system files
                if file_name.startswith('.') or file_name == 'Thumbs.db':
                    continue
                
                # For preprocessed numpy files
                if preprocessed and file_name.endswith('_points.npy'):
                    file_path = os.path.join(identity_dir, file_name)
                    identity_files.append(file_path)
                # For regular 3D files
                elif not preprocessed and file_name.endswith(('.ply', '.pcd', '.xyz')):
                    file_path = os.path.join(identity_dir, file_name)
                    identity_files.append(file_path)
            
            # Only include identities with enough samples
            if len(identity_files) >= min_samples_per_identity:
                identity_to_label[identity] = label_counter
                for file_path in identity_files:
                    file_paths.append(file_path)
                    labels.append(label_counter)
                label_counter += 1
    
    print(f"Found {len(file_paths)} files across {label_counter} identities in {directory}")
    return file_paths, labels

def prepare_data_loaders(data_dir, batch_size=16, num_points=512, num_samples_per_identity=4):
    """
    Prepare data loaders for training and validation with identity sampling
    """
    # Check if data_dir contains train and val subdirectories
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    pre_split = os.path.isdir(train_dir) and os.path.isdir(val_dir)
    
    if pre_split:
        print(f"Found pre-split train/val directories")
        # Load training data
        train_files, train_labels = collect_files_and_labels(train_dir, num_samples_per_identity)
        # Load validation data
        val_files, val_labels = collect_files_and_labels(val_dir, num_samples_per_identity)
    else:
        print(f"No pre-split directories found, creating train/val split")
        # Collect all files and labels
        file_paths, labels = collect_files_and_labels(data_dir, num_samples_per_identity)
        
        # Split into training and validation sets
        train_files, val_files, train_labels, val_labels = train_test_split(
            file_paths, labels, test_size=0.2, stratify=labels, random_state=42
        )
    
    # Check if we have enough data for triplet loss
    print(f"Training set: {len(train_files)} files, Validation set: {len(val_files)} files")
    
    # Count samples per identity to verify triplet learning is possible
    train_id_counts = {}
    for label in train_labels:
        train_id_counts[label] = train_id_counts.get(label, 0) + 1
    
    valid_ids = [id for id, count in train_id_counts.items() if count >= num_samples_per_identity]
    print(f"Identities with {num_samples_per_identity}+ samples: {len(valid_ids)}/{len(train_id_counts)}")
    
    if len(valid_ids) < 2:
        print(f"WARNING: Not enough identities with {num_samples_per_identity}+ samples for triplet loss!")
        print(f"Consider reducing num_samples_per_identity or add more data.")
    
    # Create datasets
    train_dataset = FacePointCloudDataset(
        train_files, train_labels, num_points=num_points, transform=True
    )
    val_dataset = FacePointCloudDataset(
        val_files, val_labels, num_points=num_points, transform=False
    )
    
    # Create identity sampler for training
    # Adjust batch size to be a multiple of num_samples_per_identity
    adjusted_batch_size = (batch_size // num_samples_per_identity) * num_samples_per_identity
    if adjusted_batch_size != batch_size:
        print(f"Adjusted batch size from {batch_size} to {adjusted_batch_size} to fit identity sampling")
        batch_size = adjusted_batch_size
    
    train_sampler = IdentitySampler(
        train_labels, batch_size, num_samples_per_identity=num_samples_per_identity
    )
    
    # macOS has limitations with multiprocessing
    num_workers = 0  # Use 0 to avoid multiprocessing issues on macOS
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=False
    )
    
    return train_loader, val_loader, batch_size  # Return the possibly adjusted batch_size

# Update the train_model function to accept the adjusted batch_size
def train_model(data_dir, model_save_dir, num_epochs=20, batch_size=16, 
                learning_rate=0.0005, embedding_size=128, num_points=256, 
                margin=0.2, k_neighbors=8, device_str="auto", 
                gradient_accumulation_steps=4, num_samples_per_identity=4):
    """
    Memory-optimized training function with proper identity sampling
    """
    # Create save directory if it doesn't exist
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Set device based on availability
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)
        
    print(f"Using device: {device}")
    
    # Prepare data loaders with identity sampling
    train_loader, val_loader, adjusted_batch_size = prepare_data_loaders(
        data_dir, batch_size=batch_size, num_points=num_points,
        num_samples_per_identity=num_samples_per_identity
    )
    batch_size = adjusted_batch_size  # Use the adjusted batch size
    
    # Create model
    model = FacePointCloudNetLite(embedding_size=embedding_size, 
                                k=k_neighbors, 
                                num_points=num_points)
    model = model.to(device)
    
    # Create criterion with FIXED implementation
    criterion = TripletLossMemoryEfficient(margin=margin)
    
    # Create optimizer with weight decay for regularization
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Create scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=learning_rate/10
    )
    
    # Training history
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print(f"Starting training for {num_epochs} epochs")
    print(f"Memory optimization settings:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Points per cloud: {num_points}")
    print(f"  - Embedding size: {embedding_size}")
    print(f"  - K neighbors: {k_neighbors}")
    print(f"  - Gradient accumulation steps: {gradient_accumulation_steps}")
    
    # Main training loop
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        
        # Train for one epoch with gradient accumulation
        model.train()
        total_loss = 0
        optimizer.zero_grad()  # Zero gradients at start of epoch
        
        # Progress bar
        pbar = tqdm(train_loader, desc="Training")
        
        for batch_idx, (data, labels) in enumerate(pbar):
            # Clean up memory before processing batch
            if hasattr(torch, 'mps') and device.type == 'mps':
                torch.mps.empty_cache()
                
            # Move data to device
            data, labels = data.to(device), labels.to(device)
            
            # Forward pass
            embeddings = model(data)
            
            # Calculate loss
            loss = criterion(embeddings, labels) / gradient_accumulation_steps
            
            # Backpropagate
            loss.backward()
            
            # Update weights every N steps or at the last batch
            if (batch_idx + 1) % gradient_accumulation_steps == 0 or batch_idx == len(train_loader) - 1:
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                optimizer.zero_grad()
            
            # Update statistics (use the un-scaled loss for reporting)
            batch_loss = loss.item() * gradient_accumulation_steps
            total_loss += batch_loss
            
            # Update progress bar
            pbar.set_postfix({'loss': batch_loss})
            
            # Force clean up CUDA/MPS memory after each batch
            del data, labels, embeddings, loss
            if hasattr(torch, 'mps') and device.type == 'mps':
                torch.mps.empty_cache()
            
        train_loss = total_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # Step the scheduler once per epoch (not per batch)
        scheduler.step()
        
        # Print current learning rate
        current_lr = scheduler.get_last_lr()[0]
        print(f"Current learning rate: {current_lr:.6f}")
        
        # Validate
        model.eval()
        val_loss = 0
        
        if len(val_loader) > 0:  # Only validate if we have validation data
            with torch.no_grad():
                for data, labels in tqdm(val_loader, desc="Validation"):
                    # Clean up memory
                    if hasattr(torch, 'mps') and device.type == 'mps':
                        torch.mps.empty_cache()
                        
                    # Move data to device
                    data, labels = data.to(device), labels.to(device)
                    
                    # Forward pass
                    embeddings = model(data)
                    
                    # Calculate loss
                    loss = criterion(embeddings, labels)
                    
                    # Update statistics
                    val_loss += loss.item()
                    
                    # Force clean up memory
                    del data, labels, embeddings, loss
                    if hasattr(torch, 'mps') and device.type == 'mps':
                        torch.mps.empty_cache()
            
            val_loss = val_loss / len(val_loader)
        else:
            val_loss = train_loss  # If no validation data, use training loss
            
        val_losses.append(val_loss)
        
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
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"Model saved to {checkpoint_path}")
        
        # Save latest model
        latest_path = os.path.join(model_save_dir, "latest_model.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, latest_path)
        
        # Clean up memory at the end of each epoch
        if hasattr(torch, 'mps') and device.type == 'mps':
            torch.mps.empty_cache()
    
    print("Training complete!")
    return model

def debug_train_model(data_dir, model_save_dir, num_epochs=5, batch_size=8, 
                learning_rate=0.0001, embedding_size=128, num_points=256, 
                margin=0.1, use_cpu=False):
    """
    Simplified training function for debugging
    """
    # Create save directory if it doesn't exist
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Set device - allow forcing CPU
    if use_cpu:
        device = torch.device("cpu")
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    # Prepare data - simplified approach
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    
    # Load training data
    print(f"Loading training data from {train_dir}")
    train_files, train_labels = collect_files_and_labels(train_dir, min_samples_per_identity=4)
    print(f"Found {len(train_files)} training files across {len(set(train_labels))} identities")
    
    # Create validation set from training if not available
    if not os.path.isdir(val_dir) or not os.listdir(val_dir):
        print("No validation directory or it's empty - creating validation set from training data")
        # Use 20% of training data for validation
        train_ratio = 0.8
        n_train = int(len(train_files) * train_ratio)
        
        # Ensure we maintain identity balance in split
        unique_labels = list(set(train_labels))
        print(f"Splitting {len(unique_labels)} identities between train and validation")
        
        # Simplify by just taking 1 sample from each identity for validation
        val_files = []
        val_labels = []
        kept_train_files = []
        kept_train_labels = []
        
        for label in unique_labels:
            # Get indices for this identity
            indices = [i for i, l in enumerate(train_labels) if l == label]
            
            # Take 1 sample for validation, the rest for training
            val_idx = indices[0]
            train_idx = indices[1:]
            
            val_files.append(train_files[val_idx])
            val_labels.append(train_labels[val_idx])
            
            for idx in train_idx:
                kept_train_files.append(train_files[idx])
                kept_train_labels.append(train_labels[idx])
        
        train_files = kept_train_files
        train_labels = kept_train_labels
        
        print(f"Split into {len(train_files)} training and {len(val_files)} validation files")
    else:
        # Load validation data
        val_files, val_labels = collect_files_and_labels(val_dir, min_samples_per_identity=1)
        print(f"Found {len(val_files)} validation files")
    
    # Create datasets with reduced complexity
    print("Creating datasets")
    train_dataset = FacePointCloudDataset(
        train_files, train_labels, num_points=num_points, transform=True
    )
    
    val_dataset = FacePointCloudDataset(
        val_files, val_labels, num_points=num_points, transform=False
    ) if val_files else None
    
    # Create data loaders - simplified approach
    print("Creating data loaders")
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=False
    ) if val_dataset else None
    
    # Create model
    print("Creating model")
    model = FacePointCloudNetLite(
        embedding_size=embedding_size,
        k=8,
        num_points=num_points
    )
    model = model.to(device)
    
    # Create criterion - use the fixed triplet loss
    print("Creating loss function")
    criterion = TripletLossMemoryEfficient(margin=margin)
    
    # Create optimizer
    print("Creating optimizer")
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Main training loop
    print("Starting training loop")
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        
        # Train
        model.train()
        total_loss = 0
        batch_count = 0
        
        print("Iterating through batches")
        try:
            for batch_idx, (data, labels) in enumerate(train_loader):
                print(f"Processing batch {batch_idx+1}/{len(train_loader)}")
                
                # Check batch composition
                unique_labels = torch.unique(labels)
                print(f"Batch has {len(unique_labels)} unique identities: {unique_labels.tolist()}")
                
                # Skip batches with only one identity (can't form triplets)
                if len(unique_labels) < 2:
                    print("⚠️ Skipping batch with fewer than 2 identities")
                    continue
                
                # Check if we have enough samples per identity
                valid_batch = True
                for label in unique_labels:
                    count = torch.sum(labels == label).item()
                    if count < 2:
                        print(f"⚠️ Identity {label} has only {count} samples (need at least 2)")
                        valid_batch = False
                
                if not valid_batch:
                    print("⚠️ Skipping batch with insufficient samples per identity")
                    continue
                
                # Move data to device
                try:
                    data, labels = data.to(device), labels.to(device)
                except Exception as e:
                    print(f"❌ Error moving data to device: {e}")
                    continue
                
                # Forward pass
                try:
                    embeddings = model(data)
                    print(f"Generated embeddings shape: {embeddings.shape}")
                except Exception as e:
                    print(f"❌ Error in forward pass: {e}")
                    continue
                
                # Calculate loss
                try:
                    loss = criterion(embeddings, labels)
                    print(f"Calculated loss: {loss.item():.4f}")
                except Exception as e:
                    print(f"❌ Error calculating loss: {e}")
                    continue
                
                # Backpropagate
                try:
                    loss.backward()
                except Exception as e:
                    print(f"❌ Error in backward pass: {e}")
                    continue
                
                # Update weights
                try:
                    optimizer.step()
                    optimizer.zero_grad()
                except Exception as e:
                    print(f"❌ Error updating weights: {e}")
                    continue
                
                # Update statistics
                total_loss += loss.item()
                batch_count += 1
                
                print(f"✅ Successfully processed batch {batch_idx+1}")
                
                # Force clean up memory for MPS
                if device.type == 'mps':
                    torch.mps.empty_cache()
                
                # Break early for testing
                if batch_idx >= 2:
                    print("Early stopping after 3 batches for testing")
                    break
        
        except Exception as e:
            print(f"❌ Error during training: {e}")
            import traceback
            traceback.print_exc()
        
        # Calculate average loss
        avg_loss = total_loss / max(1, batch_count)
        print(f"Epoch {epoch} training completed with average loss: {avg_loss:.4f}")
        
        # Validate - simplified for debugging
        if val_loader:
            model.eval()
            val_loss = 0
            val_count = 0
            
            print("Validating")
            with torch.no_grad():
                for batch_idx, (data, labels) in enumerate(val_loader):
                    # Skip invalid batches
                    if len(torch.unique(labels)) < 2:
                        continue
                    
                    data, labels = data.to(device), labels.to(device)
                    embeddings = model(data)
                    loss = criterion(embeddings, labels)
                    
                    val_loss += loss.item()
                    val_count += 1
                    
                    # Break early for testing
                    if batch_idx >= 2:
                        break
            
            avg_val_loss = val_loss / max(1, val_count) 
            print(f"Validation loss: {avg_val_loss:.4f}")
    
    print("Training complete!")
    
    # Save model
    save_path = os.path.join(model_save_dir, "debug_model.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
    }, save_path)
    print(f"Model saved to {save_path}")
    
    return model

def train_model_pk(data_dir, model_save_dir, num_epochs=20, p_identities=2, k_samples=2, 
              learning_rate=0.0005, embedding_size=128, num_points=256, 
              margin=0.2, k_neighbors=8, device_str="auto"):
    """
    Training function using PK Batch Sampling for triplet loss
    
    Args:
        data_dir: Directory containing point cloud data
        model_save_dir: Directory to save model checkpoints
        num_epochs: Number of training epochs
        p_identities: Number of identities per batch (P)
        k_samples: Number of samples per identity (K)
        learning_rate: Initial learning rate
        embedding_size: Size of face embedding
        num_points: Number of points to sample from each point cloud
        margin: Margin for triplet loss
        k_neighbors: Number of k-nearest neighbors for EdgeConv
        device_str: Device to train on ('auto', 'cuda', 'mps', 'cpu')
    """
    # Create save directory if it doesn't exist
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Set device based on availability
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)
    
    print(f"Using device: {device}")
    
    # Load training and validation data
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    
    print(f"Loading training data from {train_dir}")
    train_files, train_labels = collect_files_and_labels(train_dir, min_samples_per_identity=k_samples)
    print(f"Found {len(train_files)} training files across {len(set(train_labels))} identities")
    
    # Load validation data if available
    if os.path.isdir(val_dir):
        print(f"Loading validation data from {val_dir}")
        val_files, val_labels = collect_files_and_labels(val_dir, min_samples_per_identity=1)
        if len(val_files) == 0:
            print(f"No validation files found in {val_dir}, creating validation set from training")
            train_files, val_files, train_labels, val_labels = create_val_from_train(
                train_files, train_labels, k_samples)
    else:
        print(f"No validation directory found, creating validation set from training")
        train_files, val_files, train_labels, val_labels = create_val_from_train(
            train_files, train_labels, k_samples)
    
    print(f"Final split: {len(train_files)} training, {len(val_files)} validation files")
    
    # Create datasets
    print("Creating datasets")
    train_dataset = FacePointCloudDataset(
        train_files, train_labels, num_points=num_points, transform=True
    )
    val_dataset = FacePointCloudDataset(
        val_files, val_labels, num_points=num_points, transform=False
    )
    
    # Create batch samplers
    print(f"Creating PK Batch Sampler with P={p_identities}, K={k_samples}")
    train_batch_sampler = PKBatchSampler(train_labels, P=p_identities, K=k_samples)
    batch_size = p_identities * k_samples
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_sampler=train_batch_sampler,
        num_workers=0, pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=False
    )
    
    # Create model
    print(f"Creating model with embedding_size={embedding_size}, k_neighbors={k_neighbors}")
    model = FacePointCloudNetLite(embedding_size=embedding_size, 
                                k=k_neighbors, 
                                num_points=num_points)
    model = model.to(device)
    
    # Create criterion
    criterion = TripletLossMemoryEfficient(margin=margin)
    
    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Create scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=learning_rate/10
    )
    
    # Training history
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print(f"Starting training for {num_epochs} epochs")
    
    # Main training loop
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        
        # Train
        model.train()
        total_loss = 0
        batch_count = 0
        
        # Progress bar
        pbar = tqdm(train_loader, desc="Training")
        
        for batch_idx, (data, labels) in enumerate(pbar):
            # Verify batch composition
            unique_labels = torch.unique(labels)
            if len(unique_labels) < 2:
                print(f"Warning: Batch {batch_idx} has only {len(unique_labels)} identities, skipping")
                continue
                
            # Clean memory for MPS
            if hasattr(torch, 'mps') and device.type == 'mps':
                torch.mps.empty_cache()
                
            # Move data to device
            data, labels = data.to(device), labels.to(device)
            
            # Forward pass
            embeddings = model(data)
            
            # Calculate loss
            loss = criterion(embeddings, labels)
            
            # Backpropagate
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            
            # Update statistics
            batch_loss = loss.item()
            total_loss += batch_loss
            batch_count += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': batch_loss})
            
            # Clean memory
            del data, labels, embeddings, loss
            if hasattr(torch, 'mps') and device.type == 'mps':
                torch.mps.empty_cache()
        
        # Calculate average loss
        train_loss = total_loss / max(1, batch_count)
        train_losses.append(train_loss)
        
        # Step scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Print epoch summary
        print(f"Train Loss: {train_loss:.4f}, Learning rate: {current_lr:.6f}")
        
        # Validate
        model.eval()
        val_loss = 0
        val_batch_count = 0
        
        with torch.no_grad():
            for data, labels in tqdm(val_loader, desc="Validation"):
                # Skip batches with only one identity
                if len(torch.unique(labels)) < 2:
                    continue
                    
                # Clean memory
                if hasattr(torch, 'mps') and device.type == 'mps':
                    torch.mps.empty_cache()
                    
                # Move data to device
                data, labels = data.to(device), labels.to(device)
                
                # Forward pass
                embeddings = model(data)
                
                # Calculate loss
                loss = criterion(embeddings, labels)
                
                # Update statistics
                val_loss += loss.item()
                val_batch_count += 1
                
                # Clean memory
                del data, labels, embeddings, loss
                if hasattr(torch, 'mps') and device.type == 'mps':
                    torch.mps.empty_cache()
        
        # Calculate average validation loss
        avg_val_loss = val_loss / max(1, val_batch_count)
        val_losses.append(avg_val_loss)
        
        # Print validation summary
        print(f"Validation Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            
            print(f"New best validation loss: {best_val_loss:.4f}")
            
            # Save checkpoint
            checkpoint_path = os.path.join(model_save_dir, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': avg_val_loss,
            }, checkpoint_path)
            print(f"Model saved to {checkpoint_path}")
        
        # Save latest model
        latest_path = os.path.join(model_save_dir, "latest_model.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': avg_val_loss,
        }, latest_path)
    
    print("Training complete!")
    return model

def create_val_from_train(train_files, train_labels, k_samples):
    """
    Create a validation set from training data,
    ensuring training set still has enough samples per identity
    """
    # Group by identity
    identity_dict = {}
    for i, label in enumerate(train_labels):
        if label not in identity_dict:
            identity_dict[label] = []
        identity_dict[label].append(i)
    
    # Create validation set
    val_indices = []
    train_indices = []
    
    # For each identity, take 1 sample for validation, 
    # ensuring at least k_samples remain for training
    for label, indices in identity_dict.items():
        if len(indices) > k_samples:
            # Take first sample for validation
            val_indices.append(indices[0])
            # Keep the rest for training
            train_indices.extend(indices[1:])
        else:
            # If not enough samples, keep all for training
            train_indices.extend(indices)
    
    # Create new file and label lists
    new_train_files = [train_files[i] for i in train_indices]
    new_train_labels = [train_labels[i] for i in train_indices]
    val_files = [train_files[i] for i in val_indices]
    val_labels = [train_labels[i] for i in val_indices]
    
    return new_train_files, val_files, new_train_labels, val_labels

def robust_training(data_dir, model_save_dir, num_epochs=10, 
                p_identities=2, k_samples=2, embedding_size=64, 
                num_points=256, margin=0.1, force_cpu=False, 
                detect_anomaly=False):
    """
    Robust training function with fallback to CPU and error checking
    """
    # Set anomaly detection if requested (helps identify NaN issues)
    if detect_anomaly:
        print("Enabling PyTorch anomaly detection")
        torch.autograd.set_detect_anomaly(True)
    
    # Create save directory
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Set device - use CPU if forced or if MPS fails
    if force_cpu:
        device = torch.device("cpu")
        print("Forced CPU usage")
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
            try:
                # Test MPS with a small tensor
                test_tensor = torch.randn(10, 10).to("mps")
                test_result = test_tensor @ test_tensor.t()
                del test_tensor, test_result
                device = torch.device("mps")
                print("MPS test successful")
            except Exception as e:
                print(f"MPS test failed: {e}")
                print("Falling back to CPU")
                device = torch.device("cpu")
        else:
            device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    # Load training and validation data
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    
    print(f"Loading training data from {train_dir}")
    train_files, train_labels = collect_files_and_labels(train_dir, min_samples_per_identity=k_samples)
    
    print(f"Loading validation data from {val_dir}")
    val_files, val_labels = collect_files_and_labels(val_dir, min_samples_per_identity=1)
    
    print(f"Data summary:")
    print(f"  Training: {len(train_files)} files, {len(set(train_labels))} identities")
    print(f"  Validation: {len(val_files)} files, {len(set(val_labels))} identities")
    
    # Create datasets
    train_dataset = FacePointCloudDataset(
        train_files, train_labels, num_points=num_points, transform=True
    )
    val_dataset = FacePointCloudDataset(
        val_files, val_labels, num_points=num_points, transform=False
    )
    
    # Create batch sampler
    print(f"Creating batch sampler with P={p_identities}, K={k_samples}")
    train_sampler = SimplePKBatchSampler(train_labels, P=p_identities, K=k_samples)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_sampler=train_sampler,
        num_workers=0, pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=8, shuffle=False,
        num_workers=0, pin_memory=False
    )
    
    # Create model - smaller for stability
    print(f"Creating model with embedding_size={embedding_size}")
    model = FacePointCloudNetLite(
        embedding_size=embedding_size,
        k=8,
        num_points=num_points
    )
    model = model.to(device)
    
    # Create optimizer and criterion
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-5)
    criterion = TripletLossMemoryEfficient(margin=margin)
    
    # Training history
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print(f"Starting training for {num_epochs} epochs")
    
    try:
        # Training loop
        for epoch in range(1, num_epochs+1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            
            # Train
            model.train()
            total_loss = 0
            batch_count = 0
            
            # Progress bar
            for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
                try:
                    # Unpack batch
                    data, labels = batch
                    
                    # Skip invalid batches
                    if len(torch.unique(labels)) < 2:
                        print(f"Warning: Batch {batch_idx} has only {len(torch.unique(labels))} identities, skipping")
                        continue
                    
                    # Clean memory
                    if device.type == 'mps':
                        torch.mps.empty_cache()
                    
                    # Move data to device
                    data, labels = data.to(device), labels.to(device)
                    
                    # Forward pass
                    embeddings = model(data)
                    
                    # Check for NaN or Inf in embeddings
                    if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
                        print(f"Warning: NaN or Inf detected in embeddings - skipping batch {batch_idx}")
                        continue
                    
                    # Calculate loss
                    loss = criterion(embeddings, labels)
                    
                    # Check for NaN or Inf in loss
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"Warning: NaN or Inf loss detected: {loss.item()} - skipping batch {batch_idx}")
                        continue
                    
                    # Reset gradients
                    optimizer.zero_grad()
                    
                    try:
                        # Backward pass
                        loss.backward()
                        
                        # Gradient clipping for stability
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        
                        # Update weights
                        optimizer.step()
                        
                        # Update statistics
                        batch_loss = loss.item()
                        total_loss += batch_loss
                        batch_count += 1
                        
                    except RuntimeError as e:
                        print(f"Error in backward pass on batch {batch_idx}: {e}")
                        # Skip this batch but continue training
                        if "MPS backend" in str(e):
                            print("This appears to be an MPS-specific error. Consider using force_cpu=True.")
                    
                    # Clean up
                    del data, labels, embeddings, loss
                    if device.type == 'mps':
                        torch.mps.empty_cache()
                        
                except Exception as e:
                    print(f"Error processing batch {batch_idx}: {e}")
                    # Continue with next batch
            
            # Calculate average loss
            avg_train_loss = total_loss / max(1, batch_count) if batch_count > 0 else float('inf')
            train_losses.append(avg_train_loss)
            
            print(f"Train Loss: {avg_train_loss:.4f}")
            
            # Validate
            try:
                model.eval()
                total_val_loss = 0
                val_batch_count = 0
                
                with torch.no_grad():
                    for data, labels in tqdm(val_loader, desc="Validation"):
                        # Skip batches with only one identity
                        if len(torch.unique(labels)) < 2:
                            continue
                        
                        # Move data to device  
                        data, labels = data.to(device), labels.to(device)
                        
                        # Forward pass
                        embeddings = model(data)
                        
                        # Check for NaN or Inf
                        if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
                            continue
                        
                        # Calculate loss
                        loss = criterion(embeddings, labels)
                        
                        # Check for NaN or Inf
                        if torch.isnan(loss) or torch.isinf(loss):
                            continue
                        
                        # Update statistics
                        total_val_loss += loss.item()
                        val_batch_count += 1
                        
                        # Clean up
                        del data, labels, embeddings, loss
                
                # Calculate average validation loss
                avg_val_loss = total_val_loss / max(1, val_batch_count) if val_batch_count > 0 else float('inf')
                val_losses.append(avg_val_loss)
                
                print(f"Validation Loss: {avg_val_loss:.4f}")
                
                # Save best model
                if avg_val_loss < best_val_loss and not np.isinf(avg_val_loss):
                    best_val_loss = avg_val_loss
                    print(f"New best validation loss: {best_val_loss:.4f}")
                    
                    # Save model
                    checkpoint_path = os.path.join(model_save_dir, "best_model.pth")
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': avg_train_loss,
                        'val_loss': avg_val_loss,
                    }, checkpoint_path)
                    print(f"Model saved to {checkpoint_path}")
                
            except Exception as e:
                print(f"Error during validation: {e}")
                # Continue with next epoch
            
            # Save latest model in case of crashes
            if epoch % 5 == 0 or epoch == num_epochs:
                latest_path = os.path.join(model_save_dir, f"model_epoch_{epoch}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss if 'avg_val_loss' in locals() else float('inf'),
                }, latest_path)
                print(f"Checkpoint saved to {latest_path}")
    
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Training error: {e}")
        import traceback
        traceback.print_exc()
    
    print("Training complete!")
    
    # Make sure all values are valid for plotting
    train_losses = [x if not np.isnan(x) and not np.isinf(x) else 0 for x in train_losses]
    val_losses = [x if not np.isnan(x) and not np.isinf(x) else 0 for x in val_losses]
    
    return model, train_losses, val_losses

def simplified_training(data_dir, model_save_dir, num_epochs=10, 
                   p_identities=2, k_samples=2, embedding_size=64, 
                   num_points=256, margin=0.1, device_str="auto"):
    """
    Simplified training function with more efficient batch sampling
    """
    # Create save directory if it doesn't exist
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Set device
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)
    
    print(f"Using device: {device}")
    
    # Load training and validation data
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    
    print(f"Loading training data from {train_dir}")
    train_files, train_labels = collect_files_and_labels(train_dir, min_samples_per_identity=k_samples)
    
    print(f"Loading validation data from {val_dir}")
    val_files, val_labels = collect_files_and_labels(val_dir, min_samples_per_identity=1)
    
    print(f"Data summary:")
    print(f"  Training: {len(train_files)} files, {len(set(train_labels))} identities")
    print(f"  Validation: {len(val_files)} files, {len(set(val_labels))} identities")
    
    # Create datasets
    train_dataset = FacePointCloudDataset(
        train_files, train_labels, num_points=num_points, transform=True
    )
    val_dataset = FacePointCloudDataset(
        val_files, val_labels, num_points=num_points, transform=False
    )
    
    # Create simplified batch sampler
    print(f"Creating batch sampler with P={p_identities}, K={k_samples}")
    train_sampler = SimplePKBatchSampler(train_labels, P=p_identities, K=k_samples)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_sampler=train_sampler,
        num_workers=0, pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=8, shuffle=False,
        num_workers=0, pin_memory=False
    )
    
    # Create model
    print(f"Creating model with embedding_size={embedding_size}")
    model = FacePointCloudNetLite(
        embedding_size=embedding_size,
        k=8,
        num_points=num_points
    )
    model = model.to(device)
    
    # Create optimizer and criterion
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-5)
    criterion = TripletLossMemoryEfficient(margin=margin)
    
    # Training history
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print(f"Starting training for {num_epochs} epochs")
    
    # Training loop
    for epoch in range(1, num_epochs+1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        
        # Train
        model.train()
        total_loss = 0
        batch_count = 0
        
        # Progress bar
        pbar = tqdm(train_loader, desc="Training")
        
        for batch_idx, (data, labels) in enumerate(pbar):
            # Skip invalid batches (shouldn't happen with SimplePKBatchSampler)
            if len(torch.unique(labels)) < 2:
                print(f"Warning: Batch {batch_idx} has only {len(torch.unique(labels))} identities, skipping")
                continue
            
            # Clean memory for MPS
            if device.type == 'mps':
                torch.mps.empty_cache()
            
            # Move data to device
            data, labels = data.to(device), labels.to(device)
            
            # Forward pass
            embeddings = model(data)
            
            # Calculate loss
            loss = criterion(embeddings, labels)
            
            # Backpropagate
            optimizer.zero_grad()
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Update statistics
            batch_loss = loss.item()
            total_loss += batch_loss
            batch_count += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': f"{batch_loss:.4f}"})
            
            # Clean up
            del data, labels, embeddings, loss
            if device.type == 'mps':
                torch.mps.empty_cache()
        
        # Calculate average loss
        avg_train_loss = total_loss / max(1, batch_count)
        train_losses.append(avg_train_loss)
        
        print(f"Train Loss: {avg_train_loss:.4f}")
        
        # Validate
        model.eval()
        total_val_loss = 0
        val_batch_count = 0
        
        with torch.no_grad():
            for data, labels in tqdm(val_loader, desc="Validation"):
                # Skip batches with only one identity
                if len(torch.unique(labels)) < 2:
                    continue
                
                # Move data to device  
                data, labels = data.to(device), labels.to(device)
                
                # Forward pass
                embeddings = model(data)
                
                # Calculate loss
                loss = criterion(embeddings, labels)
                
                # Update statistics
                total_val_loss += loss.item()
                val_batch_count += 1
                
                # Clean up
                del data, labels, embeddings, loss
        
        # Calculate average validation loss
        avg_val_loss = total_val_loss / max(1, val_batch_count) if val_batch_count > 0 else float('inf')
        val_losses.append(avg_val_loss)
        
        print(f"Validation Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"New best validation loss: {best_val_loss:.4f}")
            
            # Save model
            checkpoint_path = os.path.join(model_save_dir, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, checkpoint_path)
            print(f"Model saved to {checkpoint_path}")
    
    print("Training complete!")
    return model, train_losses, val_losses

class PKBatchSampler(torch.utils.data.Sampler):
    """
    PK Batch Sampler:
    - P: number of identities per batch
    - K: number of samples per identity
    
    This ensures every batch has the right composition for triplet loss
    """
    def __init__(self, labels, P=2, K=2):
        """
        Args:
            labels: List of identity labels for each sample
            P: Number of identities per batch
            K: Number of samples per identity in each batch
        """
        self.labels = np.array(labels)
        self.P = P  # Number of identities per batch
        self.K = K  # Number of samples per identity
        self.batch_size = P * K
        
        # Group samples by identity
        self.identity_dict = {}
        for i, label in enumerate(self.labels):
            if label not in self.identity_dict:
                self.identity_dict[label] = []
            self.identity_dict[label].append(i)
        
        # Filter out identities with fewer than K samples
        self.valid_identities = [
            identity for identity, indices in self.identity_dict.items()
            if len(indices) >= self.K
        ]
        
        print(f"PKBatchSampler: {len(self.valid_identities)}/{len(self.identity_dict)} identities have {K}+ samples")
        if len(self.valid_identities) < self.P:
            print(f"WARNING: Only {len(self.valid_identities)} valid identities available, need at least {self.P}")
        
    def __iter__(self):
        # Create batches
        batches = []
        
        # Calculate how many batches we can create
        available_identities = self.valid_identities.copy()
        np.random.shuffle(available_identities)
        
        # Create as many batches as possible
        while len(available_identities) >= self.P:
            # Select P identities for this batch
            batch_identities = available_identities[:self.P]
            available_identities = available_identities[self.P:]
            
            # Select K samples for each identity
            batch_indices = []
            for identity in batch_identities:
                # Get all samples for this identity
                identity_samples = self.identity_dict[identity]
                
                # If we have exactly K samples, use all of them
                if len(identity_samples) == self.K:
                    selected_samples = identity_samples
                # If we have more than K samples, randomly select K
                else:
                    selected_samples = np.random.choice(
                        identity_samples, self.K, replace=False)
                
                batch_indices.extend(selected_samples)
            
            # Add batch to list of batches
            batches.append(batch_indices)
            
            # If we've used all identities, start over
            if len(available_identities) < self.P:
                available_identities = self.valid_identities.copy()
                np.random.shuffle(available_identities)
        
        # Flatten and return
        indices = []
        for _ in range(3):  # Create 3 epochs worth of batches for good measure
            np.random.shuffle(batches)
            for batch in batches:
                indices.extend(batch)
        
        return iter(indices)
    
    def __len__(self):
        # Approximate length based on valid identities
        num_batches = len(self.valid_identities) // self.P
        return num_batches * self.batch_size * 3  # 3 epochs worth
    
class SimpleIdentitySampler(torch.utils.data.Sampler):
    """
    Simpler identity sampler that just ensures each batch has the right number
    of samples per identity without complex batching logic
    """
    def __init__(self, labels, batch_size, num_samples_per_identity=4):
        self.labels = labels
        self.batch_size = batch_size
        self.num_samples_per_identity = num_samples_per_identity
        
        # Group samples by identity
        self.identity_dict = {}
        for i, label in enumerate(labels):
            if label not in self.identity_dict:
                self.identity_dict[label] = []
            self.identity_dict[label].append(i)
        
        # Filter identities with enough samples
        self.valid_identity_dict = {k: v for k, v in self.identity_dict.items() 
                                  if len(v) >= num_samples_per_identity}
        self.identities = list(self.valid_identity_dict.keys())
        
        if len(self.identities) == 0:
            raise ValueError(f"No identity has at least {num_samples_per_identity} samples")
        
        print(f"SimpleIdentitySampler: {len(self.identities)} valid identities")
        
    def __iter__(self):
        # Just create a list of indices that ensures we have enough samples 
        # from each identity, much simpler than the complex batching approach
        indices = []
        
        # Calculate how many identities we need per batch for proper triplet loss
        ids_per_batch = max(2, self.batch_size // self.num_samples_per_identity)
        
        # Ensure we use enough identities to form valid batches
        if len(self.identities) < ids_per_batch:
            print(f"WARNING: Only {len(self.identities)} identities available, need {ids_per_batch}")
            # Use what we have with repetition
        
        # Create list of balanced samples
        for _ in range(5):  # Create multiple epochs worth of data
            # Shuffle the identities each time
            id_list = self.identities.copy()
            np.random.shuffle(id_list)
            
            for identity in id_list:
                # Get samples for this identity
                id_samples = self.valid_identity_dict[identity]
                # If we have more samples than needed, randomly select
                if len(id_samples) > self.num_samples_per_identity:
                    selected = np.random.choice(id_samples, 
                                               self.num_samples_per_identity, 
                                               replace=False)
                else:
                    selected = id_samples
                
                indices.extend(selected)
        
        # Shuffle the entire list
        np.random.shuffle(indices)
        
        print(f"Created sampler with {len(indices)} indices for {len(self.labels)} total samples")
        return iter(indices)
    
    def __len__(self):
        # Return a larger number to ensure multiple epochs of data
        return len(self.labels) * 5  # 5 epochs worth of data

class SimplePKBatchSampler(torch.utils.data.BatchSampler):
    """
    Simplified PK Batch Sampler that precomputes all batches
    - P: number of identities per batch
    - K: number of samples per identity
    """
    def __init__(self, labels, P=2, K=2):
        self.labels = np.array(labels)
        self.P = P  # Number of identities per batch
        self.K = K  # Number of samples per identity
        self.batch_size = P * K
        
        # Group samples by identity
        self.identity_dict = {}
        for i, label in enumerate(self.labels):
            if label not in self.identity_dict:
                self.identity_dict[label] = []
            self.identity_dict[label].append(i)
        
        # Filter out identities with fewer than K samples
        self.valid_identities = [
            identity for identity, indices in self.identity_dict.items()
            if len(indices) >= self.K
        ]
        
        print(f"SimplePKBatchSampler: {len(self.valid_identities)}/{len(self.identity_dict)} identities have {K}+ samples")
        
        # Precompute all batches
        self.batches = self._create_batches()
        print(f"Created {len(self.batches)} batches with {self.batch_size} samples each")
    
    def _create_batches(self):
        """
        Precompute all batches to avoid on-the-fly generation
        """
        batches = []
        identities = self.valid_identities.copy()
        
        # Create enough batches for several epochs
        for _ in range(20):  # Create 20 epochs worth of batches
            # Shuffle identities for each epoch
            np.random.shuffle(identities)
            
            # Group identities into batches of size P
            for i in range(0, len(identities), self.P):
                # If we don't have enough identities for a full batch, skip
                if i + self.P > len(identities):
                    continue
                    
                # Get the next P identities
                batch_identities = identities[i:i+self.P]
                
                # Select K samples for each identity
                batch_indices = []
                for identity in batch_identities:
                    # Get all samples for this identity
                    identity_samples = self.identity_dict[identity]
                    
                    # Randomly select K samples without replacement
                    selected_samples = np.random.choice(
                        identity_samples, self.K, replace=False)
                    
                    batch_indices.extend(selected_samples)
                
                batches.append(batch_indices)
        
        return batches
    
    def __iter__(self):
        # Shuffle the batches
        indices = list(range(len(self.batches)))
        np.random.shuffle(indices)
        
        # Return batches in shuffled order
        for idx in indices:
            yield self.batches[idx]
    
    def __len__(self):
        return len(self.batches)
        
class IdentitySampler(torch.utils.data.Sampler):
    """
    Samples batches ensuring each contains multiple samples per identity
    """
    def __init__(self, labels, batch_size, num_samples_per_identity=4):
        self.labels = labels
        self.batch_size = batch_size
        self.num_samples_per_identity = num_samples_per_identity
        
        # Group samples by identity
        self.identity_dict = {}
        for i, label in enumerate(labels):
            if label not in self.identity_dict:
                self.identity_dict[label] = []
            self.identity_dict[label].append(i)
        
        # Filter identities with few samples
        self.identity_dict = {k: v for k, v in self.identity_dict.items() 
                             if len(v) >= num_samples_per_identity}
        self.identities = list(self.identity_dict.keys())
        
        if len(self.identities) == 0:
            raise ValueError(f"No identity has at least {num_samples_per_identity} samples")
    
    def __iter__(self):
        # Determine how many identities per batch
        num_identities_per_batch = self.batch_size // self.num_samples_per_identity
        
        # Create batches
        batches = []
        available_identities = self.identities.copy()
        
        while len(available_identities) >= num_identities_per_batch:
            # Randomly select identities for this batch
            batch_identities = np.random.choice(
                available_identities, 
                num_identities_per_batch, 
                replace=False
            )
            
            # Remove selected identities from available ones
            for identity in batch_identities:
                available_identities.remove(identity)
            
            # For each selected identity, choose samples
            batch_indices = []
            for identity in batch_identities:
                samples = np.random.choice(
                    self.identity_dict[identity],
                    self.num_samples_per_identity,
                    replace=False
                )
                batch_indices.extend(samples)
            
            batches.append(batch_indices)
            
            # If we've used all identities, start over
            if len(available_identities) < num_identities_per_batch:
                available_identities = self.identities.copy()
        
        # Shuffle the batches
        np.random.shuffle(batches)
        
        # Flatten and return
        return iter([i for batch in batches for i in batch])
    
    def __len__(self):
        return len(self.labels)

if __name__ == "__main__":
    # Configuration for robust training
    config = {
        "data_dir": "/Volumes/T7 Shield/ARC Images/3D synthetic - FPC/sample_npy/split",
        "model_save_dir": "./saved_models_robust",
        "num_epochs": 20,
        "p_identities": 2,       # 2 identities per batch
        "k_samples": 2,          # 2 samples per identity
        "embedding_size": 64,    # Small embedding size for stability
        "num_points": 256,       # Reduced points per cloud
        "margin": 0.1,           # Smaller margin for triplet loss
        "force_cpu": True,       # Force CPU usage to avoid MPS issues
        "detect_anomaly": False  # Set to True for detailed error tracking (slower)
    }
    
    print("Starting robust training")
    model, train_losses, val_losses = robust_training(
        data_dir=config["data_dir"],
        model_save_dir=config["model_save_dir"],
        num_epochs=config["num_epochs"],
        p_identities=config["p_identities"],
        k_samples=config["k_samples"],
        embedding_size=config["embedding_size"],
        num_points=config["num_points"],
        margin=config["margin"],
        force_cpu=config["force_cpu"],
        detect_anomaly=config["detect_anomaly"]
    )
    
    # Plot training progress if we have valid losses
    try:
        if len(train_losses) > 0:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 5))
            plt.plot(train_losses, label='Training Loss')
            plt.plot(val_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Progress')
            plt.legend()
            plt.savefig(os.path.join(config["model_save_dir"], "training_progress.png"))
            print(f"Training progress plot saved to {config['model_save_dir']}/training_progress.png")
    except Exception as e:
        print(f"Error creating training plot: {e}")
    
    print("Training complete! Final model saved.")

"""         
if __name__ == "__main__":
    # Configuration with greatly reduced memory requirements
    config = {
        "data_dir": "/Volumes/T7 Shield/ARC Images/3D synthetic - FPC/sample_npy/split",  # Can be raw data or the split directory
        "model_save_dir": "./saved_models",
        "num_epochs": 20,
        "batch_size": 16,             
        "learning_rate": 0.001,      
        "embedding_size": 256,       
        "num_points": 512,           
        "margin": 0.2,
        "k_neighbors": 8,            
        "device": "auto",
        "gradient_accumulation_steps": 4  # Accumulate gradients over 4 batches
    }
    
    # Train model with memory optimizations
    model = train_model(
        data_dir=config["data_dir"],
        model_save_dir=config["model_save_dir"],
        num_epochs=config["num_epochs"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        embedding_size=config["embedding_size"],
        num_points=config["num_points"],
        margin=config["margin"],
        k_neighbors=config["k_neighbors"],
        device_str=config["device"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"]
    ) """
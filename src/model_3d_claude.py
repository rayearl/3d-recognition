import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class EdgeConvMemoryEfficient(nn.Module):
    """
    Memory-efficient Edge Convolution Layer with MPS-compatible distance calculation
    Avoids using torch.cdist which is not fully supported by MPS backend
    """
    def __init__(self, in_channels, out_channels, k=10):
        super(EdgeConvMemoryEfficient, self).__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2)
        )
    
    def _pairwise_distances(self, x_transposed):
        """
        Compute pairwise distances for point cloud using MPS-compatible operations
        instead of torch.cdist
        
        Args:
            x_transposed: tensor of shape [B, N, D] where B is batch size, 
                         N is number of points, D is feature dimension
        
        Returns:
            Pairwise distance matrix of shape [B, N, N]
        """
        # Get batch size and number of points
        B, N, D = x_transposed.shape
        
        # Reshape for batch matrix multiplication
        x_expanded = x_transposed.unsqueeze(2)  # [B, N, 1, D]
        y_expanded = x_transposed.unsqueeze(1)  # [B, 1, N, D]
        
        # Compute squared distances using manual broadcasting
        # ||x-y||^2 = ||x||^2 + ||y||^2 - 2*<x,y>
        x_square = torch.sum(x_expanded ** 2, dim=3, keepdim=True)  # [B, N, 1, 1]
        y_square = torch.sum(y_expanded ** 2, dim=3, keepdim=True)  # [B, 1, N, 1]
        
        # Compute dot product - manual batch matmul
        dot_product = torch.sum(x_expanded * y_expanded, dim=3, keepdim=True)  # [B, N, N, 1]
        
        # Compute squared distance
        distances_squared = x_square + y_square.transpose(2, 1) - 2 * dot_product
        
        # Ensure non-negative distances
        distances_squared = F.relu(distances_squared)
        
        # Take square root for Euclidean distance
        distances = torch.sqrt(distances_squared + 1e-8)
        
        # Reshape to [B, N, N]
        return distances.squeeze(3)
    
    def forward(self, x):
        batch_size, num_dims, num_points = x.size()
        
        # Transpose for distance calculation
        x_transposed = x.transpose(1, 2)  # [B, N, D]
        
        # Compute pairwise distances using MPS-compatible operations
        dist = self._pairwise_distances(x_transposed)  # [B, N, N]
        
        # Get k nearest neighbors
        _, idx = torch.topk(-dist, k=self.k, dim=-1)  # Use negative dist for top-k smallest
        
        # Process in chunks to save memory
        chunk_size = min(128, num_points)
        edge_features = []
        
        for i in range(0, num_points, chunk_size):
            end_i = min(i + chunk_size, num_points)
            chunk_idx = idx[:, i:end_i, :]  # [B, chunk, k]
            
            # Get central points for this chunk
            central_points = x[:, :, i:end_i].unsqueeze(3)  # [B, D, chunk, 1]
            
            # Get features of nearest neighbors
            # We process each batch sample separately to save memory
            batch_features = []
            for b in range(batch_size):
                neighbor_idx = chunk_idx[b]  # [chunk, k]
                neighbor_feats = x[b, :, neighbor_idx.view(-1)].view(num_dims, end_i-i, self.k)  # [D, chunk, k]
                batch_features.append(neighbor_feats.unsqueeze(0))
            
            neighbor_features = torch.cat(batch_features, dim=0)  # [B, D, chunk, k]
            
            # Compute edge features
            chunk_edge_feature = torch.cat([
                central_points.expand(-1, -1, -1, self.k),
                neighbor_features - central_points
            ], dim=1)  # [B, 2*D, chunk, k]
            
            # Apply convolution
            chunk_edge_feature = self.conv(chunk_edge_feature)  # [B, out_C, chunk, k]
            
            # Max pooling over k neighbors
            chunk_edge_feature = chunk_edge_feature.max(dim=-1, keepdim=False)[0]  # [B, out_C, chunk]
            
            edge_features.append(chunk_edge_feature)
        
        # Combine chunks
        edge_feature = torch.cat(edge_features, dim=2)  # [B, out_C, N]
        
        return edge_feature

class DGCNNLite(nn.Module):
    """
    Memory-efficient DGCNN implementation
    - Reduced number of channels
    - Simplified architecture
    - Uses the memory-efficient EdgeConv
    """
    def __init__(self, embedding_size=128, k=10, input_channels=3):
        super(DGCNNLite, self).__init__()
        
        # Edge convolution layers
        self.edge_conv1 = EdgeConvMemoryEfficient(input_channels, 32, k=k)
        self.edge_conv2 = EdgeConvMemoryEfficient(32, 64, k=k)
        self.edge_conv3 = EdgeConvMemoryEfficient(64, 128, k=k)
        
        # Global features
        self.global_conv = nn.Sequential(
            nn.Conv1d(160, 256, kernel_size=1, bias=False),  # <-- FIXED: Changed 224 to 160
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        # Embedding layers
        self.fc1 = nn.Linear(256, embedding_size)
        self.bn1 = nn.BatchNorm1d(embedding_size)
        
        # L2 normalization for the embedding
        self.l2_norm = lambda x: F.normalize(x, p=2, dim=1)
    
    def forward(self, x):
        # Apply edge convolutions
        x1 = self.edge_conv1(x)  # [B, 32, N]
        x2 = self.edge_conv2(x1)  # [B, 64, N]
        x3 = self.edge_conv3(x2)  # [B, 128, N]
        
        # Concatenate features - using only x1, x3 to save memory
        x = torch.cat([x1, x3], dim=1)  # [B, 160, N]
        
        # Extract global features
        x = self.global_conv(x)  # [B, 256, N]
        x = x.max(dim=-1, keepdim=False)[0]  # [B, 256]
        
        # Face embedding
        x = F.leaky_relu(self.bn1(self.fc1(x)), negative_slope=0.2)
        
        # L2 normalization
        x = self.l2_norm(x)
        
        return x

class FacePointCloudNetLite(nn.Module):
    """
    Memory-optimized Face Recognition Network using Point Cloud with RGB
    - Uses DGCNNLite for more efficient processing
    - Reduced embedding size
    - Processes fewer points
    """
    def __init__(self, embedding_size=256, k=10, num_points=512):
        super(FacePointCloudNetLite, self).__init__()
        
        self.num_points = num_points
        # Separate paths for spatial and color features
        self.spatial_transform = DGCNNLite(embedding_size=embedding_size//2, k=k, input_channels=3)  # For XYZ
        self.color_transform = DGCNNLite(embedding_size=embedding_size//2, k=k, input_channels=3)    # For RGB
        
        # Final embedding layer
        self.fc = nn.Linear(embedding_size, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size)
        
        # L2 normalization for the embedding
        self.l2_norm = lambda x: F.normalize(x, p=2, dim=1)
    
    def forward(self, points):
        # points: [B, N, 6] where N is the number of points
        # First 3 channels are XYZ, next 3 are RGB
        
        B, N, _ = points.shape
        
        # Limit the number of points
        if N > self.num_points:
            # Use deterministic sampling instead of random for stability
            idx = torch.linspace(0, N-1, self.num_points).long().to(points.device)
            points = points[:, idx, :]
        
        # Split the input into spatial and color components
        xyz = points[:, :, :3].transpose(2, 1)  # [B, 3, N]
        rgb = points[:, :, 3:].transpose(2, 1)  # [B, 3, N]
        
        # Process spatial and color information separately
        spatial_features = self.spatial_transform(xyz)  # [B, embedding_size/2]
        color_features = self.color_transform(rgb)      # [B, embedding_size/2]
        
        # Concatenate the features
        combined_features = torch.cat([spatial_features, color_features], dim=1)  # [B, embedding_size]
        
        # Final embedding
        x = self.bn(self.fc(combined_features))
        
        # L2 normalization
        x = self.l2_norm(x)
        
        return x

class TripletLossMemoryEfficient(nn.Module):
    """
    Memory-efficient implementation of Triplet Loss with MPS-compatible distance calculation
    """
    def __init__(self, margin=0.2):
        super(TripletLossMemoryEfficient, self).__init__()
        self.margin = margin
    
    def _pairwise_distances(self, embeddings):
        """
        Compute pairwise distances using MPS-compatible operations
        instead of torch.cdist
        """
        # Compute squared distances using the formula:
        # ||a-b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
        
        # Get squared norm of each embedding (vector)
        # Shape: [batch_size]
        square_norm = torch.sum(embeddings ** 2, dim=1)
        
        # Compute dot product between all embeddings
        # Shape: [batch_size, batch_size]
        dot_product = torch.mm(embeddings, embeddings.t())
        
        # Compute squared distances using the formula above
        # Shape: [batch_size, batch_size]
        distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
        
        # Fix numerical issues and ensure non-negative distances
        distances = F.relu(distances)
        
        # Return square root for actual Euclidean distances
        distances = torch.sqrt(distances + 1e-8)
        return distances
    
    def forward(self, embeddings, labels):
        """
        Compute triplet loss using batch hard mining strategy
        """
        # Normalize embeddings to ensure stability
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Compute pairwise distances using MPS-compatible operations
        pairwise_dist = self._pairwise_distances(embeddings)
        
        # Get mask for valid positives (same label, not self)
        mask_anchor_positive = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1))
        # Remove self comparisons
        mask_anchor_positive = mask_anchor_positive.float() - torch.eye(labels.size(0), device=labels.device)
        mask_anchor_positive = mask_anchor_positive.clamp(min=0.0)
        
        # Get mask for valid negatives (different label)
        mask_anchor_negative = ~torch.eq(labels.unsqueeze(0), labels.unsqueeze(1))
        
        # FIXED: Proper hardest positive mining
        # For each anchor, find the hardest positive (furthest away, same class)
        # First, set distances for non-positives to 0 (we'll take max, so they'll be ignored)
        positive_dist = pairwise_dist.clone()
        positive_dist = positive_dist * mask_anchor_positive
        # Find the hardest positive (farthest distance among positives)
        hardest_positive_dist, _ = positive_dist.max(dim=1)
        
        # FIXED: Proper hardest negative mining
        # For each anchor, find the hardest negative (closest, different class)
        # First, set distances for non-negatives to a large value (we'll take min)
        negative_dist = pairwise_dist.clone()
        negative_dist[~mask_anchor_negative] = 1e6
        # Find the hardest negative (smallest distance among negatives)
        hardest_negative_dist, _ = negative_dist.min(dim=1)
        
        # Compute triplet loss
        triplet_loss = F.relu(hardest_positive_dist - hardest_negative_dist + self.margin)
        
        # Check for valid triplets
        # Handle case where no positives exist for an anchor
        zero_mask = (hardest_positive_dist == 0)
        if zero_mask.any():
            triplet_loss[zero_mask] = 0.0
        
        # Count non-zero (active) triplets
        num_active_triplets = torch.sum(triplet_loss > 1e-6).float()
        if num_active_triplets == 0:
            return torch.tensor(0.0, device=embeddings.device)
        
        return triplet_loss.mean()

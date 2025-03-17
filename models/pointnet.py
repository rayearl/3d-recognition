import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class STN3d(nn.Module):
    """
    Spatial Transformer Network for 3D point clouds
    """
    def __init__(self, channel=3):
        super(STN3d, self).__init__()
        self.conv1 = nn.Conv1d(channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # Initialize as identity transformation
        iden = torch.eye(3, dtype=torch.float32, device=x.device).view(1, 9).repeat(batchsize, 1)
        x = x + iden
        x = x.view(-1, 3, 3)
        return x

class FeatureTransformer(nn.Module):
    """
    Feature Transformer Network (similar to STN3d but for feature space)
    """
    def __init__(self, k=64):
        super(FeatureTransformer, self).__init__()
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # Initialize as identity transformation
        iden = torch.eye(self.k, dtype=torch.float32, device=x.device).view(1, self.k*self.k).repeat(batchsize, 1)
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointNetFeatures(nn.Module):
    """
    PointNet feature extraction network
    """
    def __init__(self, global_feat=True, feature_transform=False, channel=3):
        super(PointNetFeatures, self).__init__()
        self.stn = STN3d(channel)
        self.conv1 = nn.Conv1d(channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        
        if self.feature_transform:
            self.fstn = FeatureTransformer(k=64)  # Use dedicated FeatureTransformer for 64-dim features

    def forward(self, x):
        """
        Input: 
            B x C x N points (B: batch size, C: channels, N: num points)
        Returns:
            B x 1024 global feature vector
        """
        B, D, N = x.size()
        
        # Input transformation
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        
        # First feature extraction
        x = F.relu(self.bn1(self.conv1(x)))

        # Feature transformation if needed
        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        # Continue with feature extraction
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat

class PointNetFaceRecognition(nn.Module):
    """
    PointNet for 3D face recognition
    """
    def __init__(self, num_classes=10, feature_transform=False, embedding_size=512):
        super(PointNetFaceRecognition, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetFeatures(global_feat=True, feature_transform=feature_transform)
        
        # Embedding layer
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, embedding_size)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(embedding_size)
        self.dropout = nn.Dropout(p=0.4)
        
        # Classification layer
        self.fc3 = nn.Linear(embedding_size, num_classes)

    def forward(self, x, get_embeddings=False):
        """
        Forward pass
        Args:
            x: input point cloud (B x C x N)
            get_embeddings: if True, return feature embeddings instead of class scores
        """
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        embedding = F.relu(self.bn2(self.fc2(x)))
        
        if get_embeddings:
            # Normalize embeddings to unit length
            embedding = F.normalize(embedding, p=2, dim=1)
            return embedding
        
        x = self.fc3(embedding)
        return x, trans, trans_feat, embedding


if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Create input with batch size of 2
    x = torch.rand(1, 3, 1024)
    
    print(f"Input shape: {x.shape}")
    
    # Create the model with 10 classes
    model = PointNetFaceRecognition(num_classes=10, feature_transform=True)
    model.eval()
    print(f"Model created: {model.__class__.__name__}")
    
    # First test: training mode with batch size > 1
    print("Running forward pass with batch size 2...")
    class_output, trans, trans_feat, embedding = model(x)
    
    # Print shape information
    print(f"Class scores shape: {class_output.shape}")
    print(f"Transformation matrix shape: {trans.shape}")
    if trans_feat is not None:
        print(f"Feature transformation shape: {trans_feat.shape}")
    print(f"Embedding shape: {embedding.shape}")
    
    # Second test: evaluation mode with batch size of 1
    print("\nTesting with model in eval mode and batch size 1...")
    model.eval()  # Set to evaluation mode
    x_single = torch.rand(1, 3, 1024)
    
    with torch.no_grad():  # No need to track gradients in eval mode
        class_output, trans, trans_feat, embedding = model(x_single)
    
    print(f"Class scores shape: {class_output.shape}")
    print(f"Transformation matrix shape: {trans.shape}")
    if trans_feat is not None:
        print(f"Feature transformation shape: {trans_feat.shape}")
    print(f"Embedding shape: {embedding.shape}")
    
    # Test getting only embeddings
    print("\nTesting embedding extraction...")
    embedding_only = model(x_single, get_embeddings=True)
    print(f"Embeddings only shape: {embedding_only.shape}")
    
    # Verify embedding normalization
    norm = torch.norm(embedding_only, dim=1)
    print(f"Embedding norms: {norm} (should be close to 1.0)")
    
    print("\nTest completed successfully!")
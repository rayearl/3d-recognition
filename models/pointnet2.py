import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def square_distance(src, dst):
    """
    Calculate squared distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src^2,dim=-1)+sum(dst^2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, ...]
    Return:
        new_points:, indexed points data, [B, S, ..., C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


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


class PointNetSetAbstraction(nn.Module):
    """
    PointNet++ Set Abstraction Layer
    """
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all=False, use_xyz=True):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        self.use_xyz = use_xyz
        
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        
        if use_xyz:
            in_channel += 3
            
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points features, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points features, [B, D', S]
        """
        xyz = xyz.transpose(1, 2).contiguous()  # [B, N, C]
        if points is not None:
            points = points.transpose(1, 2).contiguous()  # [B, N, D]
            
        B, N, C = xyz.shape
        
        if self.group_all:
            new_xyz = torch.zeros(B, 1, C).to(xyz.device)
            grouped_xyz = xyz.view(B, 1, N, C)
            grouped_xyz_norm = grouped_xyz  # For group_all, we don't need to normalize
        else:
            fps_idx = farthest_point_sample(xyz, self.npoint)  # [B, npoint]
            new_xyz = index_points(xyz, fps_idx)  # [B, npoint, C]
            idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
            grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C]
            grouped_xyz_norm = grouped_xyz - new_xyz.view(B, self.npoint, 1, C)
            
        if points is not None:
            if self.group_all:
                grouped_points = points.view(B, 1, N, -1)
            else:
                grouped_points = index_points(points, idx)  # [B, npoint, nsample, D]
            if self.use_xyz:
                grouped_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
        else:
            grouped_points = grouped_xyz_norm
            
        # [B, npoint, nsample, C+D] -> [B, C+D, npoint, nsample]
        grouped_points = grouped_points.permute(0, 3, 1, 2).contiguous()
        
        # Apply MLPs
        for i, conv in enumerate(self.mlp_convs):
            grouped_points = F.relu(self.mlp_bns[i](conv(grouped_points)))
            
        # Max pooling
        new_points = torch.max(grouped_points, -1)[0]  # [B, D', npoint]
        
        new_xyz = new_xyz.transpose(1, 2).contiguous()
        return new_xyz, new_points


class PointNetSetAbstractionMsg(nn.Module):
    """
    PointNet++ Set Abstraction Layer with Multi-Scale Grouping
    """
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list, use_xyz=True):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.use_xyz = use_xyz
        
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            
            if use_xyz:
                last_channel = in_channel + 3
            else:
                last_channel = in_channel
                
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
                
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points features, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sampled multi-scale points features, [B, D', S]
        """
        xyz = xyz.transpose(1, 2).contiguous()  # [B, N, C]
        if points is not None:
            points = points.transpose(1, 2).contiguous()  # [B, N, D]
            
        B, N, C = xyz.shape
        
        # Sample points
        fps_idx = farthest_point_sample(xyz, self.npoint)  # [B, npoint]
        new_xyz = index_points(xyz, fps_idx)  # [B, npoint, C]
        
        new_points_list = []
        
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            # Group points for this radius
            idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, idx)  # [B, npoint, K, C]
            
            # Calculate local coordinates (normalization)
            grouped_xyz_norm = grouped_xyz - new_xyz.view(B, self.npoint, 1, C)
            
            if points is not None:
                grouped_points = index_points(points, idx)  # [B, npoint, K, D]
                if self.use_xyz:
                    grouped_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
            else:
                grouped_points = grouped_xyz_norm
                
            # Reshape for 2D convolution
            grouped_points = grouped_points.permute(0, 3, 1, 2).contiguous()  # [B, D, npoint, K]
            
            # Apply MLPs for this radius scale
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points = F.relu(bn(conv(grouped_points)))
            
            # Max pooling for this radius scale
            new_points = torch.max(grouped_points, -1)[0]  # [B, D', npoint]
            new_points_list.append(new_points)
            
        # Concatenate multi-scale features
        new_points_concat = torch.cat(new_points_list, dim=1)
        
        new_xyz = new_xyz.transpose(1, 2).contiguous()
        return new_xyz, new_points_concat


class PointNetFeatureExtractor(nn.Module):
    """
    PointNet++ feature extraction module
    """
    def __init__(self, use_xyz=True, use_msg=True, feature_transform=False):
        super(PointNetFeatureExtractor, self).__init__()
        
        self.feature_transform = feature_transform
        self.use_msg = use_msg
        
        if feature_transform:
            self.stn = STN3d(3)
            
        # Multi-scale parameter settings
        if use_msg:
            self.sa1 = PointNetSetAbstractionMsg(
                npoint=512,
                radius_list=[0.1, 0.2, 0.4],
                nsample_list=[16, 32, 64],
                in_channel=0,
                mlp_list=[[32, 32, 64], [64, 64, 128], [64, 96, 128]]
            )
            
            self.sa2 = PointNetSetAbstractionMsg(
                npoint=128,
                radius_list=[0.2, 0.4, 0.8],
                nsample_list=[32, 64, 128],
                in_channel=320,  # 64+128+128
                mlp_list=[[64, 64, 128], [128, 128, 256], [128, 128, 256]]
            )
            
            self.sa3 = PointNetSetAbstraction(
                npoint=None,
                radius=None,
                nsample=None,
                in_channel=640,  # 128+256+256
                mlp=[256, 512, 1024],
                group_all=True
            )
        else:
            # Regular set abstraction parameters
            self.sa1 = PointNetSetAbstraction(
                npoint=512,
                radius=0.2,
                nsample=32,
                in_channel=0,
                mlp=[64, 64, 128],
                group_all=False,
                use_xyz=use_xyz
            )
            
            self.sa2 = PointNetSetAbstraction(
                npoint=128,
                radius=0.4,
                nsample=64,
                in_channel=128,
                mlp=[128, 128, 256],
                group_all=False,
                use_xyz=use_xyz
            )
            
            self.sa3 = PointNetSetAbstraction(
                npoint=None,
                radius=None,
                nsample=None,
                in_channel=256,
                mlp=[256, 512, 1024],
                group_all=True,
                use_xyz=use_xyz
            )

    def forward(self, xyz):
        B, C, N = xyz.shape
        
        if self.feature_transform:
            trans = self.stn(xyz)
            xyz_transformed = torch.bmm(xyz.transpose(2, 1), trans).transpose(2, 1)
            xyz = xyz_transformed
        else:
            trans = None
        
        l1_xyz, l1_points = self.sa1(xyz, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        
        global_feat = l3_points.view(B, -1)
        
        return global_feat, trans


class PointNet2FaceRecognition(nn.Module):
    """
    PointNet++ for 3D face recognition
    """
    def __init__(self, num_classes=10, use_msg=True, feature_transform=False, embedding_size=512):
        super(PointNet2FaceRecognition, self).__init__()
        
        self.feature_transform = feature_transform
        
        # Feature extraction backbone
        self.feat = PointNetFeatureExtractor(use_xyz=True, use_msg=use_msg, feature_transform=feature_transform)
        
        # Find the output feature dimension based on use_msg setting
        feat_dim = 1024
        
        # Embedding layers
        self.fc1 = nn.Linear(feat_dim, 512)
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
        # Extract global features
        global_feat, trans = self.feat(x)
        
        # Create embedding
        x = F.relu(self.bn1(self.fc1(global_feat)))
        x = self.dropout(x)
        embedding = F.relu(self.bn2(self.fc2(x)))
        
        if get_embeddings:
            # Normalize embeddings to unit length
            embedding = F.normalize(embedding, p=2, dim=1)
            return embedding
        
        # Classification
        x = self.fc3(embedding)
        
        return x, trans, None, embedding  # trans_feat is None for compatibility


if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Create input with batch size of 2
    x = torch.rand(2, 3, 1024)
    
    print(f"Input shape: {x.shape}")
    
    # Create the model with 10 classes
    model = PointNet2FaceRecognition(num_classes=10, use_msg=True, feature_transform=True)
    print(f"Model created: {model.__class__.__name__}")
    
    # First test: training mode
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
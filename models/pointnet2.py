from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import namedtuple

# Thay thế các chức năng từ etw_pytorch_utils
def fps_subsample(xyz, npoint):
    """
    FPS (Farthest Point Sampling) đơn giản
    Lưu ý: Đây là phiên bản đơn giản, không tối ưu như triển khai CUDA
    
    Args:
        xyz: điểm đầu vào (B, N, 3)
        npoint: số lượng điểm cần lấy mẫu
    
    Returns:
        centroids: chỉ số của centroids (B, npoint)
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    
    # Lấy ngẫu nhiên một điểm làm điểm khởi tạo
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        
        # Tính khoảng cách đến centroid gần nhất
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        
        # Lấy điểm xa nhất làm centroid tiếp theo
        farthest = torch.max(distance, -1)[1]
    
    return centroids

def ball_query(radius, nsample, xyz, new_xyz):
    """
    Ball query để tìm các điểm lân cận
    
    Args:
        radius: bán kính tìm kiếm
        nsample: số lượng mẫu tối đa
        xyz: điểm đầu vào (B, N, 3)
        new_xyz: centroids (B, S, 3)
    
    Returns:
        group_idx: chỉ số nhóm (B, S, nsample)
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    
    group_idx = torch.arange(N, dtype=torch.long, device=device).view(1, 1, N).repeat(B, S, 1)
    sqrdists = square_distance(new_xyz, xyz)
    
    # Điểm nằm ngoài bán kính sẽ được đánh dấu bằng N
    group_idx[sqrdists > radius ** 2] = N
    
    # Sắp xếp theo khoảng cách tăng dần và chỉ lấy nsample điểm đầu tiên
    group_idx, _ = torch.sort(group_idx, dim=-1)
    group_first = group_idx[:, :, 0:1].repeat(1, 1, nsample)
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    
    return group_idx[:, :, :nsample]

def square_distance(src, dst):
    """
    Tính bình phương khoảng cách giữa hai tập điểm
    
    Args:
        src: điểm nguồn (B, S, 3)
        dst: điểm đích (B, N, 3)
    
    Returns:
        dist: bình phương khoảng cách (B, S, N)
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    
    return dist

def index_points(points, idx):
    """
    Lấy điểm theo chỉ số
    
    Args:
        points: điểm đầu vào (B, N, C)
        idx: chỉ số (B, S) hoặc (B, S, nsample)
    
    Returns:
        new_points: điểm được lấy (B, S, C) hoặc (B, S, nsample, C)
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    
    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    
    return new_points

# Lớp Sequential tùy chỉnh để thay thế pt_utils.Seq
class Sequential(nn.Sequential):
    def __init__(self, input_channels):
        super(Sequential, self).__init__()
        self.input_channels = input_channels
    
    def fc(self, out_channels, bn=True, activation=True):
        self.add_module(
            f'fc{len(self)}',
            nn.Linear(self.input_channels, out_channels)
        )
        
        if bn:
            self.add_module(
                f'bn{len(self) - 1}',
                nn.BatchNorm1d(out_channels)
            )
        
        if activation:
            self.add_module(
                f'relu{len(self) - 1}',
                nn.ReLU(inplace=True)
            )
        
        self.input_channels = out_channels
        return self
    
    def dropout(self, p=0.5):
        self.add_module(
            f'dropout{len(self)}',
            nn.Dropout(p=p)
        )
        return self

# Định nghĩa PointnetSAModule
class PointnetSAModule(nn.Module):
    def __init__(self, npoint=None, radius=None, nsample=None, mlp=None, use_xyz=True):
        """
        PointNet Set Abstraction Layer
        
        Args:
            npoint: số lượng điểm cần lấy mẫu
            radius: bán kính tìm kiếm các điểm lân cận
            nsample: số lượng điểm lân cận tối đa
            mlp: danh sách các kích thước của MLP
            use_xyz: có sử dụng tọa độ xyz trong các đặc trưng không
        """
        super(PointnetSAModule, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.use_xyz = use_xyz
        
        # Tạo MLP (Multi-Layer Perceptron)
        if mlp is not None:
            if use_xyz:
                mlp[0] += 3
            
            self.mlp_convs = nn.ModuleList()
            self.mlp_bns = nn.ModuleList()
            
            for i in range(len(mlp) - 1):
                self.mlp_convs.append(nn.Conv2d(mlp[i], mlp[i + 1], 1))
                self.mlp_bns.append(nn.BatchNorm2d(mlp[i + 1]))
    
    def forward(self, xyz, features=None):
        """
        Forward pass
        
        Args:
            xyz: tọa độ điểm đầu vào (B, N, 3)
            features: đặc trưng điểm đầu vào (B, C, N)
        
        Returns:
            new_xyz: tọa độ điểm mới (B, npoint, 3)
            new_features: đặc trưng mới (B, mlp[-1], npoint)
        """
        # Nếu không có điểm lấy mẫu (global feature), thực hiện max pooling
        if self.npoint is None:
            # Áp dụng MLP toàn cục
            new_features = features.unsqueeze(-1) if features is not None else None
            
            # Thêm xyz vào features nếu cần
            if self.use_xyz and new_features is not None:
                new_xyz = xyz.transpose(1, 2).unsqueeze(-1)
                new_features = torch.cat([new_xyz, new_features], dim=1)
            elif self.use_xyz:
                new_features = xyz.transpose(1, 2).unsqueeze(-1)
            
            # Áp dụng MLP
            for i, conv in enumerate(self.mlp_convs):
                new_features = conv(new_features)
                new_features = self.mlp_bns[i](new_features)
                new_features = F.relu(new_features)
            
            # Max pool
            new_features = torch.max(new_features, 2)[0]
            
            # Không thay đổi xyz khi global feature
            new_xyz = None
            
            return new_xyz, new_features
            
        else:
            # FPS để lấy mẫu centroids
            fps_idx = fps_subsample(xyz, self.npoint)
            new_xyz = index_points(xyz, fps_idx)
            
            # Ball query để nhóm các điểm
            idx = ball_query(self.radius, self.nsample, xyz, new_xyz)
            grouped_xyz = index_points(xyz, idx)
            
            # Tính tọa độ tương đối
            grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)
            
            # Nhóm các đặc trưng
            if features is not None:
                grouped_features = index_points(features.transpose(1, 2), idx)
                if self.use_xyz:
                    new_features = torch.cat([grouped_xyz_norm, grouped_features], dim=-1)
                else:
                    new_features = grouped_features
            else:
                new_features = grouped_xyz_norm
            
            # Chuyển đổi để phù hợp với Conv2d (B, C, npoint, nsample)
            new_features = new_features.permute(0, 3, 1, 2).contiguous()
            
            # Áp dụng MLP
            for i, conv in enumerate(self.mlp_convs):
                new_features = conv(new_features)
                new_features = self.mlp_bns[i](new_features)
                new_features = F.relu(new_features)
            
            # Max pool trong mỗi nhóm
            new_features = torch.max(new_features, -1)[0]
            
            return new_xyz, new_features

class Pointnet2FaceRecognition(nn.Module):
    """
    PointNet++ cho nhận diện khuôn mặt 3D
    """
    def __init__(self, num_classes, input_channels=3, use_xyz=True, embedding_size=512):
        super(Pointnet2FaceRecognition, self).__init__()
        print('Khởi tạo mạng PointNet++ cho nhận diện khuôn mặt 3D')
        
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        
        # Mô-đun Set Abstraction (SA)
        self.SA_modules = nn.ModuleList()
        
        # First SA layer - local feature with high resolution
        self.SA_modules.append(
            PointnetSAModule(
                npoint=2048,
                radius=0.01,
                nsample=64,
                mlp=[input_channels, 32, 32],
                use_xyz=use_xyz,
            )
        )
        
        # Second SA Layer - medium scale features
        self.SA_modules.append(
            PointnetSAModule(
                npoint=512,
                radius=0.12,
                nsample=64,
                mlp=[32, 64, 64],
                use_xyz=use_xyz,
            )
        )
        
        # Third SA Layer - global features
        self.SA_modules.append(
            PointnetSAModule(
                npoint=None,  # Global pooling
                radius=None,
                nsample=None,
                mlp=[64, 256, 512],
                use_xyz=use_xyz,
            )
        )
        
        # Feature embedding layer
        self.fc1 = nn.Linear(512, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(512, embedding_size)
        self.bn2 = nn.BatchNorm1d(embedding_size)
        
        # Classification layer
        self.fc3 = nn.Linear(embedding_size, num_classes)

    def _break_up_pc(self, pc):
        """
        Tách point cloud thành xyz và features
        """
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].contiguous() if pc.size(-1) > 3 else None
        
        if features is not None:
            features = features.transpose(1, 2)
            
        return xyz, features

    def forward(self, pointcloud, get_embeddings=False):
        """
        Truyền xuôi mạng
        
        Args:
            pointcloud: (B, N, 3 + input_channels) tensor
            get_embeddings: Nếu True, trả về embedding vectors thay vì điểm số lớp
            
        Returns:
            cls_scores: Điểm số phân loại
            None: Để tương thích với giao diện cũ (trans)
            None: Để tương thích với giao diện cũ (trans_feat)
            embedding: Vector embedding khuôn mặt
        """
        xyz, features = self._break_up_pc(pointcloud)
        
        # Truyền qua các lớp SA
        for module in self.SA_modules:
            xyz, features = module(xyz, features)
        
        # Lớp embedding
        x = F.relu(self.bn1(self.fc1(features.squeeze(-1))))
        x = self.dropout1(x)
        embedding = F.relu(self.bn2(self.fc2(x)))
        
        if get_embeddings:
            # Chuẩn hóa embeddings thành độ dài đơn vị
            embedding = F.normalize(embedding, p=2, dim=1)
            return embedding
        
        # Điểm số phân loại
        cls_scores = self.fc3(embedding)
        
        return cls_scores, None, None, embedding  # Trả về tương thích với giao diện cũ

# Decorator cho model_fn để sử dụng trong training
def model_fn_decorator(criterion):
    ModelReturn = namedtuple("ModelReturn", ["preds", "loss", "acc"])

    def model_fn(model, data, epoch=0, eval=False):
        with torch.set_grad_enabled(not eval):
            inputs, labels = data
            inputs = inputs.to("cuda", non_blocking=True)
            labels = labels.to("cuda", non_blocking=True)

            preds, _, _, embeddings = model(inputs)
            labels = labels.view(-1)
            loss = criterion(preds, labels)

            _, classes = torch.max(preds, -1)
            acc = (classes == labels).float().sum() / labels.numel()

            return ModelReturn(preds, loss, {"acc": acc.item(), "loss": loss.item()})

    return model_fn

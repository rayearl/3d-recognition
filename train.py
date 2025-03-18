import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import time
import json
import math
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# Import our modules
from dataset.face_dataset import get_dataloaders
from models.pointnet import PointNetFaceRecognition

class ArcMarginProduct(nn.Module):
    """
    ArcFace loss implementation 
    
    References:
    - https://arxiv.org/pdf/1801.07698.pdf
    - https://github.com/deepinsight/insightface/
    """
    def __init__(self, in_features, out_features, scale=30.0, margin=0.5, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        self.easy_margin = easy_margin
        self.cos_m = torch.cos(torch.tensor(margin))
        self.sin_m = torch.sin(torch.tensor(margin))
        self.th = torch.cos(torch.tensor(math.pi - margin))
        self.mm = torch.sin(torch.tensor(math.pi - margin)) * margin

    def forward(self, input, label=None):
        # Normalize features and weights
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        
        if label is None:
            return cosine * self.scale
            
        # For training with labels
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)
        
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        
        return output

def get_feature_transform_regularization_loss(trans):
    """
    Regularization loss for the feature transform matrix to be close to orthogonal
    """
    if trans is None:
        return 0.0
        
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :].to(trans.device)
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss

def train(args):
    # Set up logging
    exp_dir = os.path.join(args.log_dir, f'exp_{time.strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(exp_dir, exist_ok=True)
    
    # Save arguments
    with open(os.path.join(exp_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Set up tensorboard
    writer = SummaryWriter(log_dir=exp_dir)
    
    # Set device
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and not args.no_cuda else
        "mps" if torch.backends.mps.is_available() and not args.no_mps else
        "cpu"
    )
    
    print(f"Using device: {device}")
    
    # Get dataloaders
    print("Loading data...")
    train_loader, test_loader, num_classes = get_dataloaders(
        args.data_dir, 
        batch_size=args.batch_size,
        num_points=args.num_points,
        num_workers=args.num_workers
    )
    print(f"Dataset loaded with {num_classes} classes")
    
    # Create model
    print("Creating model...")
    model = PointNetFaceRecognition(
        num_classes=num_classes, 
        feature_transform=args.feature_transform,
        embedding_size=args.embedding_size
    ).to(device)
    
    # Create ArcFace metric
    metric_fc = ArcMarginProduct(
        in_features=args.embedding_size,
        out_features=num_classes,
        scale=args.arc_scale,
        margin=args.arc_margin,
        easy_margin=args.easy_margin
    ).to(device)
    
    # Optimizer
    if args.use_adam:
        optimizer = optim.Adam(
            list(model.parameters()) + list(metric_fc.parameters()),
            lr=args.lr,
            betas=(0.9, 0.999),
            weight_decay=args.weight_decay
        )
    else:
        optimizer = optim.SGD(
            list(model.parameters()) + list(metric_fc.parameters()),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay
        )
    
    # Loss functions
    criterion = nn.CrossEntropyLoss()
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # Training loop
    best_acc = 0.0
    best_model_path = os.path.join(exp_dir, 'best_model.pth')
    
    for epoch in range(args.epochs):
        print(f'Epoch {epoch+1}/{args.epochs}')
        
        # Training
        model.train()
        metric_fc.train()
        train_loss = 0.0
        train_acc = 0.0
        
        for batch_idx, data in enumerate(tqdm(train_loader, desc='Training')):
            points = data['points'].to(device)
            target = data['label'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass through PointNet to get embeddings
            _, trans, trans_feat, embeddings = model(points)
            
            # Forward pass through ArcFace
            output = metric_fc(embeddings, target)
            
            # Calculate losses
            cls_loss = criterion(output, target)
            
            # Feature transform regularization loss
            if args.feature_transform:
                trans_loss = get_feature_transform_regularization_loss(trans_feat)
                loss = cls_loss + args.lambda_t * trans_loss
            else:
                loss = cls_loss
                trans_loss = 0.0
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            _, pred_cls = torch.max(output.data, 1)
            correct = pred_cls.eq(target.data).cpu().sum().item()
            
            # Update metrics
            train_loss += loss.item()
            train_acc += correct / points.size(0)
            
            if batch_idx % args.log_interval == 0:
                writer.add_scalar('train/batch_loss', loss.item(), epoch * len(train_loader) + batch_idx)
                writer.add_scalar('train/batch_cls_loss', cls_loss.item(), epoch * len(train_loader) + batch_idx)
                writer.add_scalar('train/batch_trans_loss', trans_loss if isinstance(trans_loss, float) else trans_loss.item(), 
                                 epoch * len(train_loader) + batch_idx)
                writer.add_scalar('train/batch_acc', correct / points.size(0), epoch * len(train_loader) + batch_idx)
        
        # Calculate average metrics
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        
        # Log training metrics
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/accuracy', train_acc, epoch)
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)
        
        # Evaluate on test set
        test_loss, test_acc = evaluate(model, metric_fc, test_loader, device, criterion, args)
        
        # Log test metrics
        writer.add_scalar('test/loss', test_loss, epoch)
        writer.add_scalar('test/accuracy', test_acc, epoch)
        
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'metric_fc_state_dict': metric_fc.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': test_loss,
                'accuracy': test_acc,
            }, best_model_path)
            print(f'Best model saved with accuracy: {best_acc:.4f}')
        
        # Update learning rate
        scheduler.step()
    
    # Save final model
    final_model_path = os.path.join(exp_dir, 'final_model.pth')
    torch.save({
        'epoch': args.epochs - 1,
        'model_state_dict': model.state_dict(),
        'metric_fc_state_dict': metric_fc.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': test_loss,
        'accuracy': test_acc,
    }, final_model_path)
    
    print(f'Training completed. Best accuracy: {best_acc:.4f}')
    writer.close()
    
    return best_model_path

def evaluate(model, metric_fc, dataloader, device, criterion, args):
    """
    Evaluate the model on the given dataloader
    """
    model.eval()
    metric_fc.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in tqdm(dataloader, desc='Evaluating'):
            points = data['points'].to(device)
            target = data['label'].to(device)
            
            _, trans, trans_feat, embeddings = model(points)

            output = metric_fc(embeddings, target)  # Add target parameter            
            
            cls_loss = criterion(output, target)
            
            if args.feature_transform:
                trans_loss = get_feature_transform_regularization_loss(trans_feat)
                loss = cls_loss + args.lambda_t * trans_loss
            else:
                loss = cls_loss
            
            _, pred_cls = torch.max(output.data, 1)
            correct += pred_cls.eq(target.data).cpu().sum().item()
            total += points.size(0)
            
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy

def main():
    parser = argparse.ArgumentParser(description='3D Face Recognition with PointNet and ArcFace')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, required=True, help='Data directory')
    parser.add_argument('--num_points', type=int, default=1024, help='Number of points to sample')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--use_adam', action='store_true', help='Use Adam optimizer (default: SGD)')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--no_mps', action='store_true', help='Disable MPS')

    parser.add_argument('--feature_transform', action='store_true', help='Use feature transform')
    parser.add_argument('--embedding_size', type=int, default=512, help='Embedding size')
    
    # ArcFace parameters
    parser.add_argument('--arc_scale', type=float, default=30.0, help='ArcFace scale')
    parser.add_argument('--arc_margin', type=float, default=0.5, help='ArcFace margin')
    parser.add_argument('--easy_margin', action='store_true', help='Use easy margin in ArcFace')
    
    # Loss parameters
    parser.add_argument('--lambda_t', type=float, default=0.001, help='Weight for feature transform regularization')
    
    # Misc parameters
    parser.add_argument('--log_dir', type=str, default='logs', help='Log directory')
    parser.add_argument('--log_interval', type=int, default=10, help='Logging interval for batches')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create log directory
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Train
    best_model_path = train(args)
    print(f'Best model saved at: {best_model_path}')

if __name__ == '__main__':
    main()
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import open3d as o3d
from sklearn.model_selection import train_test_split
import glob

class FaceDataset(Dataset):
    def __init__(self, root_dir, num_points=1024, transform=None, train=True, test_size=0.2, random_state=42):
        """
        Args:
            root_dir (string): Directory with person_id folders containing ply files
            num_points (int): Number of points to sample from each point cloud
            transform (callable, optional): Optional transform to be applied on a sample
            train (bool): Whether this is training set or test set
            test_size (float): Proportion of dataset to include in the test split
            random_state (int): Random seed for reproducibility
        """
        self.root_dir = root_dir
        self.num_points = num_points
        self.transform = transform
        self.train = train
        
        # Get all PLY files
        all_files = []
        person_ids = []
        
        for idx, person_folder in enumerate(os.listdir(root_dir)):
            person_path = os.path.join(root_dir, person_folder)
            if os.path.isdir(person_path):
                for ply_file in glob.glob(os.path.join(person_path, '*.ply')):
                    all_files.append(ply_file)
                    person_ids.append(idx)
        
        # Create a mapping from person_id to class_idx
        self.unique_ids = sorted(list(set(person_ids)))
        self.id_to_class = {id: idx for idx, id in enumerate(self.unique_ids)}
        
        # Split into train and test
        if test_size > 0:
            files_train, files_test, ids_train, ids_test = train_test_split(
                all_files, person_ids, test_size=test_size, random_state=random_state, stratify=person_ids
            )
            
            if train:
                self.files = files_train
                self.person_ids = ids_train
            else:
                self.files = files_test
                self.person_ids = ids_test
        else:
            self.files = all_files
            self.person_ids = person_ids

        print(f"{'Train' if train else 'Test'} dataset has {len(self.files)} samples from {len(set(self.person_ids))} people")
            
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        ply_path = self.files[idx]
        # print(ply_path)
        person_id = self.person_ids[idx]
        class_idx = self.id_to_class[person_id]
        
        # Load the point cloud
        pcd = o3d.io.read_point_cloud(ply_path)
        
        # Convert to numpy array
        points = np.asarray(pcd.points)
        
        # Sample points if there are more than num_points
        if len(points) > self.num_points:
            choice = np.random.choice(len(points), self.num_points, replace=False)
            points = points[choice, :]
        elif len(points) < self.num_points:
            # If fewer points, duplicate some to reach num_points
            choice = np.random.choice(len(points), self.num_points - len(points), replace=True)
            points = np.vstack((points, points[choice, :]))
        
        # Normalize to unit sphere
        centroid = np.mean(points, axis=0)
        points = points - centroid
        dist = np.max(np.sqrt(np.sum(points**2, axis=1)))
        points = points / dist
        
        # Convert to tensor
        points = torch.from_numpy(points.astype(np.float32))
        
        # Add batch dimension for normal expected by PointNet
        points = points.transpose(0, 1)
        
        if self.transform:
            points = self.transform(points)
            
        return {'points': points, 'label': class_idx, 'person_id': person_id, 'path': ply_path}

def get_dataloaders(root_dir, batch_size=32, num_points=1024, num_workers=4, only_one=False):
    """
    Create train and test dataloaders
    
    Args:
        root_dir (string): Directory with person_id folders containing ply files
        batch_size (int): Batch size for training
        num_points (int): Number of points to sample from each point cloud
        num_workers (int): Number of workers for dataloader
        
    Returns:
        train_loader, test_loader, num_classes
    """
    if only_one:
        dataset = FaceDataset(root_dir, test_size=0, num_points=num_points, train=True)
        loader = DataLoader(
            dataset, 
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )
        return loader

    train_dataset = FaceDataset(root_dir, num_points=num_points, train=True)
    test_dataset = FaceDataset(root_dir, num_points=num_points, train=False)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, test_loader, len(train_dataset.unique_ids)

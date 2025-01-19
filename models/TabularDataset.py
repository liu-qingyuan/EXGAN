import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class TabularDataset(Dataset):
    """Tabular Dataset for loading numerical feature data."""
    
    def __init__(
            self,
            root: str,
            train: bool = True,
            transform = None,
            target_transform = None,
            download: bool = False,
            ir: float = 1.0,
            train_indices = None,
            test_indices = None,
            scaler = None
    ) -> None:
        """
        Args:
            root (str): Root directory containing the dataset
            train (bool): If True, creates dataset from training set
            transform: Optional transform to be applied to features
            target_transform: Optional transform to be applied to labels
            download: Ignored for tabular data
            ir (float): Imbalance ratio for training set
            train_indices: Optional indices for training set
            test_indices: Optional indices for test set
            scaler: Optional pre-fitted StandardScaler
        """
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.ir = ir
        
        # Load data
        self.data_file = os.path.join(root, 'Health', 'AI4healthcare.xlsx')
        if not os.path.exists(self.data_file):
            raise RuntimeError(f'Dataset not found at {self.data_file}')
            
        # Read data
        df = pd.read_excel(self.data_file)
        features = [c for c in df.columns if c.startswith("Feature")]
        X = df[features].copy()
        y = df["Label"].copy()
        
        # Convert to numpy arrays
        X = X.values
        y = y.values
        
        # Standardize features
        if scaler is None:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        else:
            self.scaler = scaler
            X = self.scaler.transform(X)
        
        # 使用提供的索引进行数据划分
        if train_indices is not None and test_indices is not None:
            if self.train:
                self.data = torch.FloatTensor(X[train_indices])
                self.targets = torch.LongTensor(y[train_indices])
            else:
                self.data = torch.FloatTensor(X[test_indices])
                self.targets = torch.LongTensor(y[test_indices])
        else:
            # 原来的数据划分逻辑
            if self.train:
                # Handle imbalanced data for training set
                idx_0 = np.where(y == 0)[0]
                idx_1 = np.where(y == 1)[0]
                
                # Subsample majority class based on imbalance ratio
                if ir > 1:
                    n_samples = int(len(idx_1) * ir)  # ir is majority:minority ratio
                    idx_0 = idx_0[:n_samples]
                
                # Combine indices
                indices = np.concatenate([idx_0, idx_1])
                np.random.shuffle(indices)
                
                self.data = torch.FloatTensor(X[indices])
                self.targets = torch.LongTensor(y[indices])
            else:
                # Use all data for testing
                self.data = torch.FloatTensor(X)
                self.targets = torch.LongTensor(y)
            
        print(f"{'Train' if train else 'Test'} Dataset:")
        print(f"Total samples: {len(self.data)}")
        print(f"Feature dimension: {self.data.shape[1]}")
        print(f"Class distribution: {np.bincount(self.targets)}")

        if self.train:
            # 添加高斯噪声
            noise = torch.randn_like(self.data) * 0.01
            self.data = self.data + noise
            
            # 特征遮蔽
            mask = torch.rand_like(self.data) > 0.1  # 10%概率遮蔽
            self.data = self.data * mask

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (features, target) where target is index of the target class.
        """
        features, target = self.data[index], int(self.targets[index])

        if self.transform is not None:
            features = self.transform(features)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return features, target

    def __len__(self):
        return len(self.data)
    
    def get_feature_dim(self):
        """Return the dimension of features"""
        return self.data.shape[1]
    
    def get_scaler(self):
        """Return the fitted StandardScaler"""
        return self.scaler 
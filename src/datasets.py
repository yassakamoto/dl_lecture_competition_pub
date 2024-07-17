import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint
from sklearn.preprocessing import StandardScaler    ##追加

class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data") -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        
        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."

        """
        #追加：スケーリング(標準化)  平均0、分散1に各時系列データ（self.X の各チャネルごと）に処理
        # トレーニングデータでStandardScalerをfitし、同じものをvalid、testでも利用。時系列データの最後の次元に沿ってスケーリングを行う
        self.scaler = StandardScaler()
        if split == "train":
            self.X = self.scaler.fit_transform(self.X.reshape(-1, self.X.shape[-1])).reshape(self.X.shape)
        else:
            self.X = self.scaler.transform(self.X.reshape(-1, self.X.shape[-1])).reshape(self.X.shape)
        """


    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        if hasattr(self, "y"):
            return self.X[i], self.y[i], self.subject_idxs[i]
        else:
            return self.X[i], self.subject_idxs[i]
        
    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]
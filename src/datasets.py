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


        # 前処理追加
        self.X = self.preprocess(self.X)


    #前処理のクラス定義
    def preprocess(self, X: torch.Tensor) -> torch.Tensor:
        # データをNumpy配列に変換
        X_np = X.numpy()
        
        # ベースライン補正
        baseline = X_np[:, :, :50].mean(axis=2, keepdims=True)  # 最初の50サンプルをベースラインと仮定
        X_np = X_np - baseline
        
        # フィルタリング（簡易的に高周波ノイズを除去）
        from scipy.signal import butter, lfilter
        
        def butter_lowpass(cutoff, fs, order=5):
            nyquist = 0.5 * fs
            normal_cutoff = cutoff / nyquist
            b, a = butter(order, normal_cutoff, btype='low', analog=False)
            return b, a
        
        def lowpass_filter(data, cutoff=40.0, fs=250.0, order=5):
            b, a = butter_lowpass(cutoff, fs, order=order)
            y = lfilter(b, a, data)
            return y
        
        # フィルタリングをチャンネル毎に適用
        for i in range(X_np.shape[1]):
            X_np[:, i, :] = lowpass_filter(X_np[:, i, :])
        
        # スケーリング（標準化）
        scaler = StandardScaler()
        X_np = scaler.fit_transform(X_np.reshape(-1, X_np.shape[2])).reshape(X_np.shape)
        
        # Tensorに戻す
        X = torch.tensor(X_np, dtype=X.dtype)
        
        return X

    ###以上、追加


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
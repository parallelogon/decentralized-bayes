# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 21:28:59 2022

@author: zachary jones
"""
import torch
from torch.utils.data import Dataset
import numpy as np


class LogReg(Dataset):
    def __init__(self, n_samples, dim, bias, var):
        super(LogReg, self).__init__()
        

        X1 = torch.randn(n_samples//2, dim)
        X2 = bias + np.sqrt(var)*torch.randn(n_samples//2, dim) 
        X = torch.vstack((X1,X2))
        #X -= X.mean(dim=0)
        #X/= X.norm(dim = 0)
        self.data = (X-X.min(dim = 0)[0])/(X.max(dim = 0)[0]-X.min(dim=0)[0])
        

        y1 = torch.ones(n_samples//2,1)
        self.targets = torch.vstack((y1, torch.zeros_like(y1)))
        
        self.len = n_samples
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        return(self.data[idx,:],self.targets[idx])
    
    
lr = LogReg(1000, 2, 1, 1)
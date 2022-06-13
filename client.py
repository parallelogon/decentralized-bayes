# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 14:08:39 2022

@author: Zachary Jones
"""

import torch
import torch.nn as nn
from priors import *
import numpy as np
from torch.utils.data import random_split
from copy import deepcopy
#from opacus.grad_sample import GradSampleModule

class Client(nn.Module):
    '''
    Client container object, requires arguments
    b (int): total number of possible clients
    dataset (dataset): overall dataset used e.g. mnist fashion-mnist
    indices (int tensor): indices object from random_split to subset dataset, this is to
        keep one dataset downloaded on a local machine for testing purposes.
        A deployment model would have seperate copies of the dataset in a
        different filestructure
    model (nn.Module): model object (nn.sequential or CNN)
    lr (float): learning rate for the client, all must be the same currently
    prior (prior object): prior for bayesian learning.  To perform maximum
        likelihood inference use a uniform distribution (not implemented)
    '''
    def __init__(self, b,
                 dataset,
                 indices,
                 model,
                 lr: float,
                 prior = Gaussian(0,1)):
        
        super(Client, self).__init__()
        
        self.dataset = dataset
        self.indices = indices
        self.local_time = 0
        self.neighbors = []
        self.b = b
        self.N = len(self.indices)
        self.prior = prior
        self.grad = None
        self.grad_no_noise = {}
        #self.model = GradSampleModule(model)
        self.model = model
        self.local_lr = lr
        self.neighbors = None
        self.saved_weights = []
        
    def sample(self, n):
        '''
        Parameters
        ----------
        n : int
            sample size from clients own dataset

        Returns
        -------
        Tuple of data and targets

        '''
        
        index = np.random.choice(self.indices, min(n, self.N), replace = False)
        
        return(self.dataset.data[index], self.dataset.targets[index])
    
    def calculate_local_gradient(self):
        
        '''
        Calculates the log liklihood + noise for aggregation
        '''
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                
                #param.grad_sample.add_(self.prior.grad_log(param), alpha = 1.0/self.N)
                #per_sample_norms = param.grad_sample.norm(dim = [-2,-1])
                #scale_factor = torch.minimum(torch.ones_like(per_sample_norms), 100/per_sample_norms)
                #param.grad_sample.mul_(scale_factor.unsqueeze(-1).unsqueeze(-1))
                #param.grad.data = param.grad_sample.mean(dim=0)
                
                param.grad.data.mul_(self.N)
                param.grad.add_(self.prior.grad_log(param))
                self.grad_no_noise[name] = param.grad.data.clone()
                #param.grad.mul_(torch.minimum(torch.tensor([1]), 100/param.grad.norm()))
                param.grad.data.add_(torch.randn_like(param.grad),
                                alpha = -np.sqrt(2/self.local_lr)/self.b)
    
    def forward(self, x):
        x = torch.flatten(x,1)
        return self.model(x)
    
    def step(self):
        
        if self.grad is not None:
            for name, param in self.model.named_parameters():
                param.data.add_(self.grad[name].data, alpha = -self.local_lr)
                
        self.local_time += 1
        
        
    def zero_grad(self):
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                param.grad.data.zero_()
        
        self.model.zero_grad()
        
    def sample_weights(self):
        
        self.saved_weights.append(
            {name: param.data.clone() for name, param in self.model.named_parameters()})
        
    def sample_weights_with_neighbors(self, paramlist):
        avg_params = {name: param.data.clone() for name, param in self.model.named_parameters()}

        neighbors = self.neighbors[0]
        weights = self.neighbors[1]
        
        if len(neighbors) > 0:
            for name in avg_params.keys():
                for i in range(len(paramlist)):
                    avg_params[name] += paramlist[i][name].data.clone()
                avg_params[name] /= len(paramlist)
            
        self.saved_weights.append(
            {name: avg_params[name].data.clone() for name in avg_params.keys()})
        
def init_clients(dataset, n_clients, model, lr, prior = Gaussian(0,1)):
    N = len(dataset)
    s = np.sort(np.random.choice(N, n_clients-1, replace = False))
    sizes = np.diff(np.array([0] + list(s) + [N]))
    
    assert sum(sizes) == N

    datasets = random_split(dataset, sizes)

    models = [Client(n_clients, dataset, subset.indices, deepcopy(model), lr, prior)
              for subset in datasets ]
    
    return(models)

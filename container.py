# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 11:37:12 2022

@author: Zachary Jones
"""

import torch.nn as nn
import torch
import numpy as np
class Container(nn.Module):
    
    def __init__(self, client_list, loss_criterion):
        
        super(Container, self).__init__()
        
        self.clients = nn.ModuleList(client_list)
        self.loss_criterion = loss_criterion
        self.b = len(self.clients)
        
        self.random_neighbors(.5)
        
        self.reset_parameters()
        self.fanin = 0
        
    def reset_parameters(self):
        for client in self.clients:
            for name, parameter in client.model.named_parameters():
                
                if 'weight' in name:
                    self.fanin = parameter.shape[-1]
                    
                    parameter.data = 1/np.sqrt(self.fanin)*torch.randn_like(parameter) #nn.init.xavier_normal_(parameter)
                    
                else:
                    parameter.data = 1/np.sqrt(self.fanin)*torch.randn_like(parameter)#/= np.sqrt(len(parameter.data))
                    
    def forward(self, x):
        y_i = []
        for client in self.clients:
            y_i.append(client.forward(x))
        
        return torch.stack(y_i).mean(dim = 0)
    
    def loss(self, n):
        loss = 0
        for client in self.clients:
            data_i, targets_i = client.sample(n)
            y_i = client(data_i.to(torch.float))
            loss += self.loss_criterion(y_i, targets_i)
            
        return loss/self.b
            
    def random_neighbors(self, p):
        b = self.b
        graph = torch.rand(b,b) 
        graph = 0.5*(graph + graph.T)
        graph = 1.0*(graph <= p)
        vert = 1/graph.sum(dim = 0).view(-1,1)
        vert[vert.isnan()] = 0
        vert[vert.isinf()] = 0
        
        graph = torch.min(vert, vert.T)*graph
        
        graph -= graph.diag().diag()

        for i, client in enumerate(self.clients):
            client.neighbors = (torch.where(graph[i,:] != 0)[0], graph[i,graph[i,:]!=0])
        
    def backward(self, loss):
        
        loss.backward()
        
        for client in self.clients:
            client.calculate_local_gradient()
            
    def aggregate_gradients(self, p):
        
        self.random_neighbors(p)
        
        for client in self.clients:
            
            
            neighbors = client.neighbors[0]
            weights = client.neighbors[1]
            
            N_neighbors = len(neighbors)
            
            if N_neighbors > 0:
                factor = self.b/N_neighbors
                client.grad = {name:param.grad.data*factor for name, param in client.model.named_parameters()}
                
                
                for weight,neighbor in zip(weights,neighbors):
                    for name, param in self.clients[neighbor].model.named_parameters():
                        if name in list(client.grad.keys()):
                            client.grad[name].data.add_(param.grad.data, alpha = factor)                            
                            client.grad_no_noise[name].data.add_(self.clients[neighbor].grad_no_noise[name].clone(), alpha = weight*factor)
                            
            else:
                client.grad = {name:param.grad.data*self.b for name, param in client.model.named_parameters()}

    def step(self):
        for client in self.clients:
            client.step()
            
    def zero_grad(self):
        for client in self.clients:
            client.zero_grad()
            
    def train(self, n, p):
            
        self.zero_grad()
        loss = self.loss(n)
        self.backward(loss)
        self.aggregate_gradients(p)
        self.step()
        
        return loss.data
    
    def sample(self):
        for client in self.clients:
            client.sample_weights()
            
    def aggregate_sample(self):
        for client in self.clients:
            neighbors = client.neighbors[0]
            weights = client.neighbors[1]
            
            N_neighbors = len(neighbors)
            
            if N_neighbors > 0:
                list_of_params = []
                for neighbor in neighbors:
                    list_of_params.append({name: param for name, param in self.clients[neighbor].model.named_parameters()})

            client.sample_weights_with_neighbors(list_of_params)
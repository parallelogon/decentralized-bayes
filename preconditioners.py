# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 22:49:17 2022

@author: Zachary Jones
"""

from math import sqrt
from scipy.linalg import fractional_matrix_power
import torch

class Preconditioner():
  def __init__(self, d1, d2):

    if d2:
      self.type = 'weight'
      self.input_size = d2

    else:
      self.input_size = 1
      self.type = "bias"

    self.output_size = d1
    self.C = torch.eye(d1)
    self.A = torch.eye(d1)


  def update(self, grad):
    pass

  def to(self, device):
    self.C = self.C.to(device)
    self.A = self.A.to(device)
    self.device = device
    return self

# Euclidean preconditioner for use in gradient step does not change the
# distribution of the weights.  
class Euclidean(Preconditioner):
  def __init__(self, d1, d2 = None):
    super(Euclidean, self).__init__(d1, d2)

  def apply(self):
    return self.C

  def sample(self):
    if self.type == 'weight':
      sample = torch.randn(self.output_size, self.input_size).to(self.device)
    else:
      sample = torch.randn(self.output_size).to(self.device)

    # use torch.einsum instead of distinguishing between bias and weight terms,
    # not particularly slow.
    return(torch.einsum("ij,j... -> i...",self.A, sample))


# RMSProp preconditioner corresponds to giving each weight a seperate learning
# rate
class RMSProp(Preconditioner):
  def __init__(self, d1, d2 = None, decay_rate = .5, regularizer = 1e-7):
    super(RMSProp, self).__init__(d1, d2)
    self.decay_rate = decay_rate
    self.regularizer = regularizer

  @torch.no_grad()
  def update(self,grad):
    if self.type == "weight":
      g_sq = torch.einsum("ij,ij->i",grad,grad).diag()

    else:
      g_sq = torch.mul(grad,grad)

    self.C.mul_(1.0 - self.decay_rate).add_(g_sq, alpha = self.decay_rate)

    local_time = 1./(self.decay_rate**2) + 1.0
    self.decay_rate = 1.0/sqrt(local_time)

  def apply(self):
    return torch.pow(self.C + self.regularizer, 0.5)

  def sample(self):
    if self.type == 'weight':
      sample = torch.randn(self.output_size, self.input_size).to(self.device)
    else:
      sample = torch.randn(self.output_size).to(self.device)

    return(torch.einsum("ij,j... -> i...",torch.pow(self.C + self.regularizer, \
                                                    0.25), sample))

  def to(self, device):
    self.A = self.A.to(device)
    self.C = self.C.to(device)
    self.device = device
    self.regularizer = torch.tensor([self.regularizer]).to(device)
    return(self)


# inverting the fisher information matrix is extremely espensive, restrict to
# smaller models.  Fisher information gives the curvature around the gradient
# estimator and yields Amari's natural gradient
class Fisher(Preconditioner):
  def __init__(self, d1, d2 = None, decay_rate = .5, regularizer = 1e-2):
    super(Fisher, self).__init__(d1, d2)
    self.decay_rate = decay_rate
    self.regularizer = regularizer

  @torch.no_grad()
  def update(self, grad):
    self.C.mul_(1.0 - self.decay_rate)
    self.C.add_(torch.matmul(grad, grad.T),alpha = self.decay_rate)

    local_time = 1.0/self.decay_rate + 1.0
    self.decay_rate = 1.0/sqrt(local_time)

  def apply(self):
    return torch.linalg.inv(torch.add(self.C,
                                      torch.eye(self.output_size).to(self.device),
                                      alpha = self.regularizer))

  def sample(self):
    if self.type == "weight":
      sample = torch.randn(self.output_size, self.input_size).to(self.device)
    else:
      sample = torch.randn(self.output_size).to(self.device)
    
    pre = torch.add(self.C, torch.eye(self.output_size).to(device), \
                    alpha = self.regularizer)
    
    # find a better matrix square root algo this is slow and cumbersome
    return torch.linalg.solve(
        torch.from_numpy(
            fractional_matrix_power(
                pre.cpu(), 0.5)).to(torch.float).to(self.device), sample)

  def to(self, device):
    self.device = device
    self.A = self.A.to(device)
    self.C = self.C.to(device)
    return(self)

# The quasi diagonal cholesky decomposition of the inverse fisher information 
# agrees with the inverse fisher information matrix on the diagonals and the 
# first row.  Use this as an approximation of the fisher info for larger models.
class QDC(Preconditioner):
  def __init__(self, d1, d2 = None, decay_rate = .5, regularizer = 1e-2):
    super(QDC,self).__init__(d1,d2)
    self.decay_rate = decay_rate
    self.regularizer = regularizer

  @torch.no_grad()
  def update(self, grad):
    self.C.mul_(1.0 - self.decay_rate)

    if self.type == "weight":
      g_row = torch.matmul(grad[0,:], grad.T)
      g_diag = torch.einsum("ij,ji -> i", grad, grad.T).diag()

    else:
      grad_unsqueezed = torch.unsqueeze(grad,-1)
      g_row = torch.mul(grad[0], grad_unsqueezed.T)
      g_diag = torch.mul(grad, grad).diag()

    g_factor = g_diag
    g_factor[0,:] = g_row
    self.C.add_(g_factor, alpha = self.decay_rate)

    local_time = 1.0/self.decay_rate + 1.0
    self.decay_rate = 1.0/sqrt(local_time)

    a00 = torch.add(self.C[0,0], self.regularizer).pow(-0.5)
    self.A[0,0] = a00
    for i in range(1,self.output_size):

      self.A[i,i] = torch.pow(
          torch.add(
              self.C[i,i],torch.mul(
                  a00,self.C[0,i]).pow(2), alpha = -1.0 ).add_(self.regularizer),-0.5)
      
      self.A[0,i] = torch.mul(a00, self.A[i,i].pow(2)).mul_(self.C[0,i])

  def apply(self):
    return self.A.matmul(self.A.T)

  def sample(self):
    if self.type == "weight":
      sample = torch.randn(self.output_size, self.input_size).to(self.device)
    else:
      sample = torch.randn(self.output_size).to(self.device)

    return(torch.einsum("ij,j... -> i...", self.A, sample))

  def to(self, device):
    self.device = device
    self.A = self.A.to(device)
    self.C = self.C.to(device)
    return(self)
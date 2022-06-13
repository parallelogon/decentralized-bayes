# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 22:50:22 2022

@author: 
"""

import torch
from torch.optim.optimizer import Optimizer, required
from math import sqrt

class SLGD(Optimizer):
  def __init__(self, params, 
               lr=required, 
               prior = required, 
               N = required, 
               preconditioner = required,
               MALA = False):

    if lr is not required and lr < 0.0:
      raise ValueError("Invalid learning rate: {}".format(lr))

    if prior is not required and prior.__class__.__name__ not in \
    ["Gaussian", "InverseGamma"]:
      raise ValueError("Invalid choice of prior: {}".format(prior))

    if N is not required and N < 0:
      raise ValueError("Need the size of the dataset")

    if preconditioner is not required and preconditioner.__name__ not in\
    ["Euclidean","RMSProp","Fisher","QDC"]:
      raise ValueError("Invalid choice of preconditioner: {}".format(preconditioner))

    else:
      preconditioner = preconditioner

    defaults = dict(lr = lr, prior = prior, N = N,
                    preconditioner = preconditioner, MALA = MALA)
    
    super(SLGD, self).__init__(params, defaults)

    # initalizes the preconditioner
    
    for group in self.param_groups:
      p_group = []
      for p in group["params"]:
        p_group.append(preconditioner(*p.shape).to(p.device))
      group['preconditioner'] = p_group

    self.param_groups[0]["prior"].to(self.param_groups[0]["params"][0].device)

  def __setstate__(self, state):
    super(SLGD, self).__setstate__(state)

  @torch.no_grad()
  def step(self, closure=None):

    loss = None
    if closure is not None:
      with torch.enable_grad():
        loss = closure()

    for group in self.param_groups:
      lr = group['lr']
      prior = group['prior']
      N = group['N']
      preconditioner = group['preconditioner']
      MALA = group["MALA"]

      # SGD update modified
      for i,p in enumerate(group['params']):
        if p.grad is not None:
          preconditioner[i].update(p.grad)

          d_p = torch.einsum("ij,j...-> i...", preconditioner[i].apply(), 
                             p.grad.add_(prior.grad_log(p), alpha = 1.0/N))
          
          p_proposed = p.add(d_p, alpha = -lr)

          noise = preconditioner[i].sample()
          
          p.grad = d_p
          
          p_proposed.add_(noise, alpha = sqrt(2.0*lr/N))

          if not MALA:
            p.data = p_proposed.data

          else:
              raise NotImplementedError()

    return loss
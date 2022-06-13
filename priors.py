# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 14:37:34 2022

@author: Zachary Jones
"""
import torch

class Prior():
  def __init__(self, model = "gaussian", **kwargs):
    '''
    Indedpendent of the form, each instantation of the 'Prior' class keeps
    track of the model used, (gaussian etc.) and checks for parameters.
    Each particular instantation of a model must include a 'gradLog' function
    which returns the distribution specific gradient of the NEGATIVE log
    liklihood.

    This is used as part of the gradient calculation for bayesian learning
    '''
    self.model = model
    self.params = kwargs.get("params", None)

    if self.params == None:
      raise ValueError("Something has gone horribly wrong: invalid parameters")

  def __name__(self):
    return self.__class__.__name__
    
  def to(self,device):
    self.device = device


      
class Gaussian(Prior):
  def __init__(self, mean = 0.0, sig = 0.1):
    super(Gaussian, self).__init__("gaussian", params = {"mean": mean, "sig": sig})
    self.mean = torch.tensor([mean])
    self.sig = torch.tensor([sig])

  def prior(self, theta):

    logPrior = torch.norm(theta - self.mean)
    logPrior=-logPrior*logPrior/(2*self.sig**2)\
    -.5*torch.log(2*torch.pi*self.sig**2)*len(theta)

    gradLogPrior = (theta - self.mean)/self.sig**2

    self.logPrior = logPrior
    self.gradLogPrior = gradLogPrior

    return(logPrior, gradLogPrior)


  # optimize the negative log liklihood later in the model, so this is the grad
  # of the negative log prior
  def grad_log(self, theta):
    return (theta - self.mean)/self.sig**2

  def to(self,device):
    self.device = device
    self.mean = self.mean.to(device)
    self.sig = self.sig.to(device)


# Assuming normally distributed weights this is the conjugate prior.
class InverseGamma(Prior):
  def __init__(self, alpha = 0.01, beta = 0.01):
    super(InverseGamma, self).__init__("inv_gamma", params = {"alpha" : alpha,
                                                              "beta": beta})
    self.alpha = alpha
    self.beta = beta

  def __name__(self):
    return "InverseGamma"

  def prior(self, theta):
    logPrior = self.alpha * torch.log(self.beta) - (self.alpha + 1) \
    * torch.log(theta) - self.beta/theta - torch.lgamma(self.alpha)

    gradLogPrior = -(self.alpha + 1)/theta + self.beta/theta**2
    return(logPrior, gradLogPrior)
    
  def grad_log(self, theta):
    gradLogPrior = -(self.alpha + 1)/theta + self.beta/theta**2
    return gradLogPrior

  def to(self,device):
    self.device = device
    self.alpha = torch.tensor([self.alpha]).to(device)
    self.beta = torch.tensor([self.beta]).to(device)
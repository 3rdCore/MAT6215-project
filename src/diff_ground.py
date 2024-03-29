# -*- coding: utf-8 -*-
"""Diff_ground.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1u9LTQdj8lrzQutKh_55yAzJ0tk2ncIs6
"""

import numpy as np
from scipy.spatial import distance_matrix
from sklearn.manifold import MDS
from scipy import spatial
class Diff_ground():
  K = None
  K_s = None
  q_i_s = None
  K1_s = None
  D = None
  P_s = None
  G = None
  pi = None
  def __init__(self, X, epsilon, t_max, w=0.5, beta=1.0, mode = 'degree') -> None:
    self.X = X
    self.epsilon = epsilon
    self.t_max = t_max
    self.w = w #Short term weight
    self.beta = beta #long term weight
    self.mode = mode
  def fit(self):
    self.K = spatial.distance.pdist(self.X, metric='euclidean')
    self.K = spatial.distance.squareform(self.K)
    self.K_e = np.exp(-(self.K**2) / self.epsilon)
    self.q_i = 1./np.sum(self.K_e, axis = 0) # D_inverse (i.e. 1/q(x_i))
    self.M1_e = np.diag(self.q_i) @ self.K_e @ np.diag(self.q_i)
    self.D = np.sum(self.M1_e, axis = 0)
    self.P_e = np.diag(1./self.D) @ self.M1_e 

  def compute_dist(self):
    G = np.zeros((self.X.shape[0], self.X.shape[0]))
    G = distance_matrix(self.P_e,self.P_e,1)
    self.P_e = self.P_e @ self.P_e 
    for t in range(1,self.t_max):
      G = G + self.w * distance_matrix(self.P_e,self.P_e,1)
      self.w = self.w*self.w
      self.P_e = self.P_e @ self.P_e 
    vol = np.sum(self.D)
    self.pi = self.D/vol
    
    if self.mode == 'degree':
      self.G = G + self.beta * distance_matrix(self.D[:,None],self.D[:,None],1)
    if self.mode == 'stationary':
      self.G = G + self.beta * distance_matrix(self.pi[:,None],self.pi[:,None],1)
    return self.G

  def fit_transform(self):
    self.fit()
    self.compute_dist()
    embedding = MDS(n_components=2) #Multidimentional scaling
    return embedding.fit_transform(self.G)
import numpy as np

class RBModel:
  def __init__(self, mass, I, g=9.8):
    self.mass = mass
    self.I = I
    self.g = g

class IdentityModel(RBModel):
  def __init__(self):
    super().__init__(1, np.eye(3))

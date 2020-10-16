import numpy as np

class ForceDisturbance:
  def apply(self, model):
    pass

  def get(self, state, control):
    return np.hstack((self.force(state, control), np.zeros(3)))

class TorqueDisturbance:
  def apply(self, model):
    pass

  def get(self, state, control):
    return np.hstack((np.zeros(3), self.torque(state, control)))

class ModelDisturbance:
  def get(self, state, control):
    return np.zeros(6)

class LinearDrag(ForceDisturbance):
  def __init__(self, scale):
    self.c = scale

  def force(self, state, control):
    return -self.c * state.vel

class ConstantForce(ForceDisturbance):
  def __init__(self, scale):
    self.c = scale

  def force(self, state, control):
    return self.c

class MassDisturbance(ModelDisturbance):
  def __init__(self, scale):
    self.c = scale

  def apply(self, model):
    model.mass *= self.c

class InertiaDisturbance(ModelDisturbance):
  """ Assume diagonal inertia """
  def __init__(self, scales):
    self.scales = np.array(scales)

  def apply(self, model):
    assert np.allclose(model.I, np.diag(np.diag(model.I)))

    model.I[0] *= self.scales[0]
    model.I[1] *= self.scales[1]
    model.I[2] *= self.scales[2]

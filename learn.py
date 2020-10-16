import numpy as np

from python_utils.mathu import e3

class AccelLearner:
  def __init__(self, model, learner):
    self.lastt = 0
    self.lastvel = np.zeros(3)
    self.last_acc = model.g * e3
    self.gvec = -model.g * e3
    self.mass = model.mass

    self.learner = learner

    self.reset_trial()
    self.reset_data()

  def reset_data(self):
    self.xs = []
    self.ys = []
    self.trained = False

  def reset_trial(self):
    self.first = True

  def add(self, t, state, control):
    dt = t - self.lastt
    self.lastt = t

    dvel = state.vel - self.lastvel
    self.lastvel = state.vel

    accel_base = self.last_acc + self.gvec
    z = state.rot.apply(e3)
    self.last_acc = control[0] * z / self.mass

    if self.first:
      self.first = False
      return

    accel_true = dvel / dt
    accel_err = accel_true - accel_base

    # Use velocity as feature vector for now
    self.xs.append(self.get_input(t, state, control))
    self.ys.append(accel_err)

  def train(self):
    self.trained = True
    self.learner.train(np.array(self.xs), np.array(self.ys))

  def get_input(self, t, state, control):
    return state.vel

  def testpoint(self, t, state, control):
    if not self.trained:
      return np.zeros(3)

    xs = np.array([self.get_input(t, state, control)])
    return self.learner.test(xs)[0]

  def dvel(self, t, state, control):
    if not self.trained:
      return np.zeros(3)

    return self.learner.gradient(np.array([self.get_input(t, state, control)]))

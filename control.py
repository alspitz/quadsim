import numpy as np

from quadsim.learn import AccelLearner

class Controller:
  def endtrial(self):
    pass

class ControllerLearnAccel:
  def __init__(self, model, learner):
    self.accel_learner = AccelLearner(model, learner)

  def endtrial(self):
    self.accel_learner.train()
    self.accel_learner.reset_trial()

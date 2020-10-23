import numpy as np

from quadsim.learn import AccelLearner

class Controller:
  def __init__(self, output_rpm=False):
    self.output_rpm = output_rpm
    self.vars = {}

  def endtrial(self):
    pass

  def out(self, force, torque):
    if self.output_rpm:
      # This is quite hacky.
      self.vars.update(forcedes=force, torquedes=torque)
      return self.model.rpm_from_forcetorque(force, torque)

    return force, torque

class ControllerLearnAccel(Controller):
  def __init__(self, model, learner):
    super().__init__()
    self.accel_learner = AccelLearner(model, learner)

  def endtrial(self):
    self.accel_learner.train()
    self.accel_learner.reset_trial()

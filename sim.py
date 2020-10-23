import numpy as np

from scipy.spatial.transform import Rotation as R

from python_utils.timeseriesu import TimeSeries
from python_utils.mathu import e3

from quadsim.rigid_body import RigidBody

class QuadSim:
  def __init__(self, model):
    self.gvec = np.array((0, 0, -model.g))
    self.mass = model.mass
    self.rb = RigidBody(mass=model.mass, inertia=model.I)
    self.model = model

    self.force_limit = 200 # Newtons (N)
    # L_2 limit on vector
    self.torque_limit = 100 # Nm

  def setstate(self, state):
    self.rb.setstate(state)

  def forcetorque_from_u(self, u, **kwargs):
    return u

  def reset(self):
    pass

  def simulate(self, dt, t_end, controller, dists=None):
    if dists is None:
      dists = []

    ts = TimeSeries()
    n_steps = int(round(t_end / dt)) - 1

    self.reset()
    for i in range(n_steps):
      t = i * dt
      state = self.rb.state()
      bodyz_force, torque = self.forcetorque_from_u(controller.response(t, state), dt=dt)

      if bodyz_force < 0 or bodyz_force > self.force_limit:
        print("Clipping force!")
        bodyz_force = np.clip(bodyz_force, 0, self.force_limit)

      torque_norm = np.linalg.norm(torque)
      if torque_norm > self.torque_limit:
        print("Clipping torque!")
        torque *= self.torque_limit / torque_norm

      controlvars = {}
      if hasattr(controller, 'vars'):
        controlvars.update(controller.vars)

      ts.add_point(time=t, **state.__dict__, force=bodyz_force, torque=torque, **controlvars)

      force_world = bodyz_force * state.rot.apply(e3) + self.mass * self.gvec
      torque_body = torque

      for dist in dists:
        d = dist.get(state, (bodyz_force, torque))
        force_world += d[:3]
        torque_body += d[3:]

      self.rb.step(dt, force=force_world, torque=torque_body)

    ts.finalize()
    return ts

class QuadSimMotors(QuadSim):
  def __init__(self, model):
    super().__init__(model)
    self.reset()

  def reset(self):
    self.started = False

  def forcetorque_from_u(self, desrpm, dt):
    if not self.started:
      self.started = True
      self.actrpm = desrpm.copy()

    else:
      alpha = self.model.motor_tc * dt
      self.actrpm = alpha * desrpm + (1 - alpha) * self.actrpm

    return self.model.forcetorque_from_rpm(self.actrpm)

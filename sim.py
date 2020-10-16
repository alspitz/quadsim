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

  def setstate(self, state):
    self.rb.setstate(state)

  def simulate(self, dt, t_end, controller, dists=None):
    if dists is None:
      dists = []

    ts = TimeSeries()
    n_steps = int(round(t_end / dt)) - 1

    for i in range(n_steps):
      t = i * dt
      state = self.rb.state()
      bodyz_force, torque = controller(t, state)

      if bodyz_force < 0:
        print("Force is negative! Clipping...")
        bodyz_force = 0

      ts.add_point(time=t, **state.__dict__, force=bodyz_force, torque=torque)

      force_world = bodyz_force * state.rot.apply(e3) + self.mass * self.gvec
      torque_body = torque

      for dist in dists:
        d = dist.get(state, (bodyz_force, torque))
        force_world += d[:3]
        torque_body += d[3:]

      self.rb.step(dt, force=force_world, torque=torque_body)

    ts.finalize()
    return ts

import time

import numpy as np

from scipy.spatial.transform import Rotation as R

from python_utils.timeseriesu import TimeSeries
from python_utils.mathu import e3

from quadsim.rigid_body import RigidBody
from quadsim.visualizer import Vis

class QuadSim:
  def __init__(self, model, force_limit=200, torque_limit=50):
    self.gvec = np.array((0, 0, -model.g))
    self.mass = model.mass
    self.rb = RigidBody(mass=model.mass, inertia=model.I)
    self.model = model

    self.force_limit = force_limit # Newtons (N)
    # L_2 limit on vector
    self.torque_limit = torque_limit # Nm

    self.vis = Vis()

  def setstate(self, state):
    self.rb.setstate(state)

  def forcetorque_from_u(self, u, **kwargs):
    return u

  def reset(self):
    pass

  def simulate(self, dt, t_end, controller, dists=None, vis=True):
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
        #print("Clipping force!", bodyz_force)
        bodyz_force = np.clip(bodyz_force, 0, self.force_limit)

      torque_norm = np.linalg.norm(torque)
      if torque_norm > self.torque_limit:
        print("Clipping torque!", torque)
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

      if vis:
        quat = state.rot.as_quat()
        self.vis.set_state(state.pos.copy(), [quat[3], quat[0], quat[1], quat[2]])

      time.sleep(dt)

    ts.finalize()
    return ts

class QuadSimMotors(QuadSim):
  def __init__(self, model, **kwargs):
    super().__init__(model, **kwargs)
    self.reset()

  def reset(self):
    self.actrpm = self.model.rpm_from_forcetorque(self.model.mass * self.model.g, np.zeros(3))

  def forcetorque_from_u(self, desrpm, dt):
    alpha = self.model.motor_tc * dt
    self.actrpm = alpha * desrpm + (1 - alpha) * self.actrpm
    #self.actrpm = (alpha * desrpm[0] + (1 - alpha) * self.actrpm[0], alpha * desrpm[1] + (1 - alpha) * self.actrpm[1])

    #u = self.model.forcetorque_from_rpm(self.actrpm)
    #return u[0], self.model.forcetorque_from_rpm(desrpm)[1]
    return self.model.forcetorque_from_rpm(self.actrpm)
    #return self.actrpm

import numpy as np

from scipy.spatial.transform import Rotation as R

import rot_metrics

from python_utils.mathu import normang, e1, e2, e3

from quadsim.control import Controller, ControllerLearnAccel
from quadsim.flatness import get_xdot_xddot

class FBLinController(Controller):
  def __init__(self, model, dt):
    super().__init__()

    self.Kpos = 6 * 120 * np.eye(3)
    self.Kvel = 4 * 120 * np.eye(3)
    self.Kacc = 120 * np.eye(3)
    self.Kjerk = 16 * np.eye(3)

    self.Kyaw = 30
    self.Kyawvel = 10

    self.gvec = np.array((0, 0, -model.g))

    self.model = model
    self.dt = dt

    self.u = model.g
    self.udot = 0

  def response(self, t, state):
    ang_world = state.rot.apply(state.ang)

    x = state.rot.apply(e1)
    y = state.rot.apply(e2)
    z = state.rot.apply(e3)
    xdot = np.cross(ang_world, x)
    zdot = np.cross(ang_world, z)

    acc = self.u * z + self.gvec
    jerk = self.udot * z + self.u * zdot

    pos_error = state.pos - self.ref.pos(t)
    vel_error = state.vel - self.ref.vel(t)
    acc_error = acc - self.ref.acc(t)
    jerk_error = jerk - self.ref.jerk(t)

    snap = -self.Kpos.dot(pos_error) - self.Kvel.dot(vel_error) - self.Kacc.dot(acc_error) - self.Kjerk.dot(jerk_error) + self.ref.snap(t)

    uddot = snap.dot(z) + self.u * zdot.dot(zdot)
    zddot = (1.0 / self.u) * (snap - 2 * self.udot * zdot - uddot * z)
    angaccel_world = np.cross(z, zddot - np.cross(ang_world, zdot))

    # Convert to body frame.
    angaccel = state.rot.inv().apply(angaccel_world)

    yaw = np.arctan2(x[1], x[0])

    x_xy_norm = x[0] ** 2 + x[1] ** 2

    # Perhaps this is too limiting.
    # Should still include fblin terms in this case
    # should only turn off "yaw feedback".
    if x_xy_norm > 1e-8:
      yawvel = (-x[1] * xdot[0] + x[0] * xdot[1]) / x_xy_norm
      yawacc = -self.Kyaw * normang(yaw - self.ref.yaw(t)) - self.Kyawvel * (yawvel - self.ref.yawvel(t)) + self.ref.yawacc(t)

      _, xddot = get_xdot_xddot(yawvel, yawacc, x, z, zdot, zddot)
      alpha_cross_x = xddot - np.cross(ang_world, xdot)
      # See notes for proof of below line: "Angular Velocity for Yaw" in Notability.
      angaccel[2] = alpha_cross_x.dot(y)

    bodyz_force = self.model.mass * self.u
    torque = self.model.I.dot(angaccel) + np.cross(state.ang, self.model.I.dot(state.ang))

    # This assumes this controller is only called once every dt
    self.u += self.udot * self.dt
    self.udot += uddot * self.dt

    self.vars.update(uddot=uddot)

    return self.out(bodyz_force, torque)

class FBLinControllerLearnAccel(ControllerLearnAccel):
  def __init__(self, model, learner, dt):
    self.Kpos = 6 * 120 * np.eye(3)
    self.Kvel = 4 * 120 * np.eye(3)
    self.Kacc = 120 * np.eye(3)
    self.Kjerk = 16 * np.eye(3)

    self.Kyaw = 30
    self.Kyawvel = 10

    self.gvec = np.array((0, 0, -model.g))

    self.model = model
    self.dt = dt

    self.reset()
    super().__init__(model, learner)

  def reset(self):
    self.u = self.model.g
    self.udot = 0

  def endtrial(self):
    self.reset()
    super().endtrial()

  def response(self, t, state):
    ang_world = state.rot.apply(state.ang)

    x = state.rot.apply(e1)
    y = state.rot.apply(e2)
    z = state.rot.apply(e3)
    xdot = np.cross(ang_world, x)
    zdot = np.cross(ang_world, z)

    control_for_learner = None
    acc_error = self.accel_learner.testpoint(t, state, control_for_learner)
    dvel = self.accel_learner.dvel(t, state, control_for_learner)

    acc = self.u * z + self.gvec + acc_error
    jerk = self.udot * z + self.u * zdot + dvel.dot(acc)

    pos_error = state.pos - self.ref.pos(t)
    vel_error = state.vel - self.ref.vel(t)
    acc_error = acc - self.ref.acc(t)
    jerk_error = jerk - self.ref.jerk(t)

    snap = -self.Kpos.dot(pos_error) - self.Kvel.dot(vel_error) - self.Kacc.dot(acc_error) - self.Kjerk.dot(jerk_error) + self.ref.snap(t)
    snap -= dvel.dot(jerk)

    uddot = snap.dot(z) + self.u * zdot.dot(zdot)
    zddot = (1.0 / self.u) * (snap - 2 * self.udot * zdot - uddot * z)
    angaccel_world = np.cross(z, zddot - np.cross(ang_world, zdot))

    # Convert to body frame.
    angaccel = state.rot.inv().apply(angaccel_world)

    yaw = np.arctan2(x[1], x[0])
    yawvel = (-x[1] * xdot[0] + x[0] * xdot[1]) / (x[0] ** 2 + x[1] ** 2)
    yawacc = -self.Kyaw * normang(yaw - self.ref.yaw(t)) - self.Kyawvel * (yawvel - self.ref.yawvel(t)) + self.ref.yawacc(t)

    _, xddot = get_xdot_xddot(yawvel, yawacc, x, z, zdot, zddot)
    alpha_cross_x = xddot - np.cross(ang_world, xdot)
    # See notes for proof of below line: "Angular Velocity for Yaw" in Notability.
    angaccel[2] = alpha_cross_x.dot(y)

    bodyz_force = self.model.mass * self.u
    torque = self.model.I.dot(angaccel) + np.cross(state.ang, self.model.I.dot(state.ang))

    # This assumes this controller is only called once every dt
    self.u += self.udot * self.dt
    self.udot += uddot * self.dt

    self.accel_learner.add(t, state, (bodyz_force, torque))

    return self.out(bodyz_force, torque)

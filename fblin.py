import numpy as np

from scipy.spatial.transform import Rotation as R

import rot_metrics

from python_utils.mathu import normang, e1, e2, e3

from quadsim.control import Controller, ControllerLearnAccel, torque_from_aa
from quadsim.flatness import (
  get_xdot_xddot,
  a_from_z,
  j_from_zdot,
  yaw_zyx_from_x,
  yawdot_zyx_from_xdot,
  alpha_from_zddot,
  alpha_from_flat,
  uddot_from_s,
  zddot_from_s,
  omega_from_zdot,
  omega_from_flat,
)

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

    # Linear control in flat snap space.
    acc = a_from_z(z, self.u)
    jerk = j_from_zdot(z, self.u, self.udot, zdot)

    pos_error = state.pos - self.ref.pos(t)
    vel_error = state.vel - self.ref.vel(t)
    acc_error = acc - self.ref.acc(t)
    jerk_error = jerk - self.ref.jerk(t)

    snap = -self.Kpos.dot(pos_error) - self.Kvel.dot(vel_error) - self.Kacc.dot(acc_error) - self.Kjerk.dot(jerk_error) + self.ref.snap(t)

    # Linear control in flat yaw space.
    yaw = yaw_zyx_from_x(x)
    yawdot = yawdot_zyx_from_xdot(x, xdot)

    yaw_error = normang(yaw - self.ref.yaw(t))
    yawdot_error = yawdot - self.ref.yawvel(t)

    yawacc = -self.Kyaw * yaw_error - self.Kyawvel * yawdot_error + self.ref.yawacc(t)

    angaccel_world = alpha_from_flat(self.u, acc, jerk, snap, yaw, yawdot, yawacc)

    # Needed?
    uddot = uddot_from_s(self.u, snap, z, zdot)

    # Convert to body frame.
    angaccel = state.rot.inv().apply(angaccel_world)

    bodyz_force = self.model.mass * self.u
    torque = torque_from_aa(angaccel, self.model.I, state.ang)

    # This assumes this controller is only called once every dt
    self.u += self.udot * self.dt
    self.udot += uddot * self.dt

    self.vars.update(uddot=uddot)

    return self.out(bodyz_force, torque)

class FBLinControllerLearnAccel(ControllerLearnAccel):
  def __init__(self, model, learner, features, dt):
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
    super().__init__(model, learner, features)

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
    dpos = self.accel_learner.dpos(t, state, control_for_learner)
    dvel = self.accel_learner.dvel(t, state, control_for_learner)

    acc = self.u * z + self.gvec + acc_error
    aed1 = dpos.dot(state.vel) + dvel.dot(acc)

    jerk = self.udot * z + self.u * zdot + aed1
    aed2 = dpos.dot(acc) + dvel.dot(jerk)

    pos_error = state.pos - self.ref.pos(t)
    vel_error = state.vel - self.ref.vel(t)
    acc_error = acc - self.ref.acc(t)
    jerk_error = jerk - self.ref.jerk(t)

    snap = -self.Kpos.dot(pos_error) - self.Kvel.dot(vel_error) - self.Kacc.dot(acc_error) - self.Kjerk.dot(jerk_error) + self.ref.snap(t)
    snap -= aed2

    uddot = snap.dot(z) + self.u * zdot.dot(zdot)
    zddot = (1.0 / self.u) * (snap - 2 * self.udot * zdot - uddot * z)
    angaccel_world = np.cross(z, zddot - np.cross(ang_world, zdot))

    # Convert to body frame.
    angaccel = state.rot.inv().apply(angaccel_world)

    yaw = np.arctan2(x[1], x[0])
    yawvel = (-x[1] * xdot[0] + x[0] * xdot[1]) / (x[0] ** 2 + x[1] ** 2)
    yawacc = -self.Kyaw * normang(yaw - self.ref.yaw(t)) - self.Kyawvel * (yawvel - self.ref.yawvel(t)) + self.ref.yawacc(t)

    _, xddot = get_xdot_xddot(yawvel, yawacc, state.rot, zdot, zddot)
    alpha_cross_x = xddot - np.cross(ang_world, xdot)
    # See notes for proof of below line: "Angular Velocity for Yaw" in Notability.
    angaccel[2] = alpha_cross_x.dot(y)

    bodyz_force = self.model.mass * self.u
    torque = self.model.I.dot(angaccel) + np.cross(state.ang, self.model.I.dot(state.ang))

    # This assumes this controller is only called once every dt
    self.u += self.udot * self.dt
    self.udot += uddot * self.dt

    self.add_datapoint(t, state, (bodyz_force, torque))

    return self.out(bodyz_force, torque)

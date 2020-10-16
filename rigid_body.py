import numpy as np

from scipy.spatial.transform import Rotation as R

from python_utils.mathu import quat_mult, vector_quat, normalized

class State:
  def __init__(self, pos=np.zeros(3), vel=np.zeros(3), rot=R.identity(), ang=np.zeros(3)):
    self.pos = pos
    self.vel = vel
    self.rot = rot
    self.ang = ang

class RigidBody:
  def __init__(self, mass=1, inertia=np.eye(3)):
    self.mass = mass
    self.I = inertia
    self.Iinv = np.linalg.inv(self.I)
    self.setstate(State())

  def setstate(self, state):
    """
        pos and vel are in the fixed frame
        rot transforms from the body frame to the fixed frame.
        ang is in the body frame
    """
    self.pos = state.pos.copy()
    self.vel = state.vel.copy()
    quat_wlast = state.rot.as_quat()
    self.quat = np.hstack((quat_wlast[3], quat_wlast[0:3]))
    self.ang = state.ang.copy()

  def step(self, dt, force, torque):
    """
        force is in the fixed frame
        torque is in the body frame
    """
    accel = force / self.mass
    # Euler equation:
    # Torque = I alpha + om x (I om)
    alpha = self.Iinv.dot(torque - np.cross(self.ang, self.I.dot(self.ang)))

    self.pos += self.vel * dt + 0.5 * accel * dt ** 2
    self.vel += accel * dt

    quat_deriv = quat_mult(self.quat, vector_quat(self.ang)) / 2.0
    quat_dd = quat_mult(self.quat, vector_quat(alpha)) / 2.0

    self.quat += quat_deriv * dt + 0.5 * quat_dd * dt ** 2
    self.quat = normalized(self.quat)
    self.ang += alpha * dt

  def getpos(self):
    return self.pos.copy()
  def getvel(self):
    return self.vel.copy()
  def getrot(self):
    return R.from_quat(np.hstack((self.quat[1:4], self.quat[0])))
  def getang(self):
    return self.ang.copy()

  def state(self):
    return State(self.getpos(), self.getvel(), self.getrot(), self.getang())

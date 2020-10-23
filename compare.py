import copy

import numpy as np

from scipy.spatial.transform import Rotation as R

from quadsim.cascaded import CascadedController
from quadsim.dist import MassDisturbance, LinearDrag, InertiaDisturbance, MotorModelDisturbance
from quadsim.fblin import FBLinController
from quadsim.flatref import StaticRef, PosLineYawLine, YawLine, PosLine
from quadsim.models import IdentityModel, rocky09
from quadsim.rigid_body import State
from quadsim.sim import QuadSim, QuadSimMotors

from python_utils.plotu import subplot, set_3daxes_equal

import rot_metrics

import matplotlib.pyplot as plt

class Test:
  def __init__(self, controller, n_trials=1, **plotargs):
    self.controller = controller
    self.plotargs = plotargs
    self.n_trials = n_trials
    self.results = []

def run(model, startstate, ref, dists, tests, dt, t_end, sim_motors=False):
  for dist in dists:
    dist.apply(model)

  if sim_motors:
    quadsim = QuadSimMotors(model)
  else:
    quadsim = QuadSim(model)

  for test in tests:
    print(test.plotargs['label'])

    for i in range(test.n_trials):
      quadsim.setstate(startstate)
      test.controller.ref = ref

      if sim_motors:
        test.controller.output_rpm = True

      ts = quadsim.simulate(dt=dt, t_end=t_end, controller=test.controller, dists=dists)

      if not i:
        test.posdes = ref.pos(ts.times).T
        test.veldes = ref.vel(ts.times).T
        test.yawdes = ref.yaw(ts.times)

      ts.poserr = ts.pos - test.posdes
      ts.poserrnorm = np.linalg.norm(ts.poserr, axis=1)

      test.results.append(ts)
      test.controller.endtrial()

      print(f"\tTrial {i + 1} mean err:", np.mean(ts.poserrnorm))

def plot(tests, ref):
  fig = plt.figure(num="Trajectory")
  ax = fig.add_subplot(111, projection='3d')
  ax.set_title("Trajectory")
  ax.set_xlabel("X (m)")
  ax.set_ylabel("Y (m)")
  ax.set_zlabel("Z (m)")

  desargs = dict(label="Desired", linestyle='dashed', color='black')

  for i, test in enumerate(tests):
    plotargs = test.plotargs

    # Use last trial... Temporary TODO
    ts = test.results[-1]

    if len(test.results) > 1:
      print(f"WARNING: {plotargs['label']} has {len(test.results)} trials. Only plotting last.")

    if not i:
      subplot(ts.times, test.posdes, yname="Pos. (m)", title="Position", **desargs)
      subplot(ts.times, test.veldes, yname="Vel. (m)", title="Velocity", **desargs)
      plt.figure("ZYX Euler Angles")
      plt.subplot(313)
      plt.plot(ts.times, test.yawdes, **desargs)

    # Pos Vel
    subplot(ts.times, ts.pos, yname="Pos. (m)", title="Position", **plotargs)
    subplot(ts.times, ts.vel, yname="Vel. (m)", title="Velocity", **plotargs)

    # Euler
    eulers = np.array([rot.as_euler('ZYX')[::-1] for rot in ts.rot])
    subplot(ts.times, eulers, yname="Euler (rad)", title="ZYX Euler Angles", **plotargs)

    # Angvel Torque
    subplot(ts.times, ts.ang, yname="$\\omega$ (rad/s)", title="Angular Velocity", **plotargs)
    subplot(ts.times, ts.torque, yname="Torque (Nm)", title="Control Torque", **plotargs)
    if hasattr(ts, 'torquedes'):
      subplot(ts.times, ts.torquedes, yname="Torque (Nm)", title="Control Torque", **plotargs, linestyle='dashed')

    # Thrust
    subplot(ts.times, ts.force, yname="Thrust (N)", title="Control Thrust", **plotargs)
    if hasattr(ts, 'forcedes'):
      subplot(ts.times, ts.forcedes, yname="Thrust (N)", title="Control Thrust", **plotargs, linestyle='dashed')

    if hasattr(ts, 'uddot'):
      subplot(ts.times, ts.uddot, yname="u ddot (m/s$^4$)", title="U ddot", **plotargs)

    # Pos Err
    subplot(ts.times, ts.poserr,  yname="Pos. Err. (m)", title="Position Error Per Axis", **plotargs)
    subplot(ts.times, ts.poserrnorm, yname="Position Error (m)", title="Position Error", **plotargs)

    # 3D Traj
    ax.plot(ts.pos[:, 0], ts.pos[:, 1], ts.pos[:, 2], **plotargs)
    set_3daxes_equal(ax)

  plt.show()

if __name__ == "__main__":
  edgesize = 1.0

  startpos = np.zeros(3)
  endpos = np.array((edgesize, edgesize, 0.0))
  startyaw = 0.0
  #endyaw = np.pi / 2
  endyaw = 0.0
  duration = 2.0

  startstate = State(
    pos=startpos,
    vel=np.zeros(3),
    rot=R.from_euler('ZYX', [startyaw, 0.0, 0.0]),
    ang=np.zeros(3)
  )

  dt = 0.002
  t_end = 1.5
  sim_motors = True

  ref = StaticRef(pos=endpos, yaw=endyaw)
  #ref = PosLine(start=startpos, end=endpos, yaw=endyaw, duration=duration)
  #ref = YawLine(pos=endpos, startyaw=startyaw, endyaw=endyaw, duration=duration)
  #ref = PosLineYawLine(start=startpos, end=endpos, startyaw=startyaw, endyaw=endyaw, duration=duration)

  dists = [
    #MassDisturbance(1.2),
    #InertiaDisturbance((1.3, 1.2, 1.5)),
    #LinearDrag(2.2),
    MotorModelDisturbance(0.8)
  ]

  #model_control = IdentityModel()
  model_control = rocky09()
  model_true = copy.deepcopy(model_control)

  def casc(rm):
    return CascadedController(model_control, rot_metric=rm)

  fblin = FBLinController(model_control, dt=dt)

  tests = [
    Test(casc(rot_metrics.euler_zyx), label="Euler ZYX"),
    Test(casc(rot_metrics.rotvec_tilt_priority2), label="Rotation Vector TP"),
    Test(fblin, label="FBLin")
  ]

  run(model_true, startstate, ref, dists, tests, dt, t_end, sim_motors=sim_motors)
  plot(tests, ref)

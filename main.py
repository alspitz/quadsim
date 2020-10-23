import numpy as np

from quadsim.sim import QuadSim
from quadsim.cascaded import CascadedController
from quadsim.fblin import FBLinController
from quadsim.flatref import StaticRef
from quadsim.models import IdentityModel

from python_utils.plotu import subplot, set_3daxes_equal

#import rot_metrics

import matplotlib.pyplot as plt

if __name__ == "__main__":
  posdes = np.array((3, 3, 0.0))
  yawdes = np.pi / 2
  #yawdes = 0.0
  dt = 0.005

  ref = StaticRef(pos=posdes, yaw=yawdes)

  model = IdentityModel()

  quadsim = QuadSim(model)
  #controller = CascadedController(model, ref=ref, rot_metric=rot_metrics.rotvec_tilt_priority2)
  #controller = CascadedController(model, ref=ref, rot_metric=rot_metrics.euler_zyx)
  controller = FBLinController(model, dt=dt)

  controller.ref = ref
  ts = quadsim.simulate(dt=dt, t_end=5.0, controller=controller)

  eulers = np.array([rot.as_euler('ZYX')[::-1] for rot in ts.rot])

  subplot(ts.times, ts.pos, yname="Pos. (m)", title="Position")
  subplot(ts.times, ts.vel, yname="Vel. (m)", title="Velocity")
  subplot(ts.times, eulers, yname="Euler (rad)", title="ZYX Euler Angles")
  subplot(ts.times, ts.ang, yname="$\\omega$ (rad/s)", title="Angular Velocity")

  fig = plt.figure(num="Trajectory")
  ax = fig.add_subplot(111, projection='3d')
  plt.plot(ts.pos[:, 0], ts.pos[:, 1], ts.pos[:, 2])
  plt.xlabel("X (m)")
  plt.ylabel("Y (m)")
  ax.set_zlabel("Z (m)")
  plt.title("Trajectory")
  set_3daxes_equal(ax)

  plt.show()

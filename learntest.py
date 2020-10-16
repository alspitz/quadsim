import copy

import numpy as np

from scipy.spatial.transform import Rotation as R

from quadsim.cascaded import CascadedController, CascadedControllerLearnAccel
from quadsim.dist import MassDisturbance, LinearDrag, InertiaDisturbance
from quadsim.fblin import FBLinController, FBLinControllerLearnAccel
from quadsim.flatref import StaticRef, PosLineYawLine, YawLine, PosLine
from quadsim.models import IdentityModel
from quadsim.rigid_body import State

from quadsim.compare import Test, plot, run

import rot_metrics

from regression import Linear

if __name__ == "__main__":
  startpos = np.zeros(3)
  endpos = np.array((3, 3, 0.0))
  startyaw = 0.0
  endyaw = np.pi / 2
  duration = 2.0

  startstate = State(
    pos=startpos,
    vel=np.zeros(3),
    rot=R.from_euler('ZYX', [startyaw, 0.0, 0.0]),
    ang=np.zeros(3)
  )

  dt = 0.005
  t_end = 3.0

  n_trials = 6

  #ref = StaticRef(pos=endpos, yaw=endyaw)
  #ref = PosLine(start=startpos, end=endpos, yaw=endyaw, duration=duration)
  #ref = YawLine(pos=endpos, startyaw=startyaw, endyaw=endyaw, duration=duration)
  ref = PosLineYawLine(start=startpos, end=endpos, startyaw=startyaw, endyaw=endyaw, duration=duration)

  dists = [
    #MassDisturbance(1.2),
    #InertiaDisturbance((1.3, 1.2, 1.5)),
    LinearDrag(2.2),
  ]

  model_control = IdentityModel()
  model_true = copy.deepcopy(model_control)

  learner = Linear()

  def casc(rm):
    return CascadedController(model_control, rot_metric=rm)

  fblin_base = FBLinController(model_control, dt=dt)
  fblin_learn = FBLinControllerLearnAccel(model_control, learner, dt=dt)

  tests = [
    Test(casc(rot_metrics.rotvec_tilt_priority2), label="Baseline FF"),
    Test(CascadedControllerLearnAccel(model_control, learner, rot_metric=rot_metrics.rotvec_tilt_priority2), label="FFLin Learn", n_trials=n_trials),
    Test(fblin_base, label="Baseline FB"),
    Test(fblin_learn, label="FBLin Learn", n_trials=n_trials)
  ]

  run(model_true, startstate, ref, dists, tests, dt, t_end)
  plot(tests, ref)

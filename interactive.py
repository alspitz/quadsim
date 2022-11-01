import sys
import time

import numpy as np

from scipy.spatial.transform import Rotation as R

from quadsim.sim import QuadSim
from quadsim.cascaded import CascadedController
from quadsim.flatref import StaticRef
from quadsim.models import IdentityModel

import quadsim.rot_metrics as rot_metrics

import atexit
import select
import termios
import tty
class KeyGrabber:
  def __init__(self):
    self.setup_keys()
    atexit.register(self.restore_keys)

  def setup_keys(self):
    fd = sys.stdin.fileno()
    self.old_settings = termios.tcgetattr(fd)
    tty.setcbreak(fd)

  def restore_keys(self):
    termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self.old_settings)

  def read(self):
    chars = []
    while select.select([sys.stdin,], [], [], 0.0)[0]:
      chars.append(sys.stdin.read(1))
    return chars

keymap = dict(
    left='q',
    right='e',
    forward='w',
    backward='s',
    up='=',
    down='-',
    turnleft='a',
    turnright='d',
)

if __name__ == "__main__":
  dt = 0.005
  vis = True

  ref = StaticRef(np.zeros(3))

  model = IdentityModel()
  quadsim = QuadSim(model)

  controller = CascadedController(model, rot_metric=rot_metrics.rotvec_tilt_priority2)
  controller.ref = ref

  kg = KeyGrabber()

  i = 0
  while 1:
    quadsim.step(i * dt, dt, controller)

    if vis:
      state = quadsim.rb.state()
      quadsim.vis.set_state(state.pos.copy(), state.rot)
      time.sleep(0.3 * dt)

    i += 1

    movedist = 0.25
    yawdist = np.pi / 8

    chars = kg.read()
    if chars:
      for c in chars:
        bodymove = np.zeros(3)
        if c == keymap['forward']:
          bodymove[0] += movedist
        elif c == keymap['left']:
          bodymove[1] += movedist
        elif c == keymap['backward']:
          bodymove[0] -= movedist
        elif c == keymap['right']:
          bodymove[1] -= movedist
        elif c == keymap['up']:
          bodymove[2] += movedist
        elif c == keymap['down']:
          bodymove[2] -= movedist
        elif c == keymap['turnleft']:
          ref.yawdes += yawdist
        elif c == keymap['turnright']:
          ref.yawdes -= yawdist
        else:
          print(c)

        bodyrot = R.from_euler('ZYX', [ref.yawdes, 0, 0])
        ref.posdes += bodyrot.apply(bodymove)

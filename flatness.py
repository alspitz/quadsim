import numpy as np

def get_xdot_xddot(yawvel, yawacc, x, z, zdot, zddot):
  """ Return (xdot, xddot) in world frame (same frame as args?)
      for Euler ZYX
  """
  # Solve for angvel z using three constraints
  # (1) x2 / x1 = tan(yaw)
  # (2) x dot z = 0
  # (3) x dot x = 1

  x_xy_norm2 = x[0] ** 2 + x[1] ** 2

  A = np.zeros((3, 3))
  A[0, :] = np.array((-x[1], x[0], 0)) / x_xy_norm2
  A[1, :] = z
  A[2, :] = x

  Ainv = np.linalg.inv(A)

  b = np.array((yawvel, -x.dot(zdot), 0))
  xdot = Ainv.dot(b)

  # Solve for xddot using the same three constraints
  # It turns out that the A matrix is the same.
  b20 = yawacc + 2 * ((-x[1] * xdot[0] + x[0] * xdot[1]) * (x[0] * xdot[0] + x[1] * xdot[1])) / (x_xy_norm2 ** 2)
  b2 = np.array((b20, -x.dot(zddot) -2 * xdot.dot(zdot), -xdot.dot(xdot)))
  xddot = Ainv.dot(b2)

  return (xdot, xddot)

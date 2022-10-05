import numpy as np

import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf

class Vis:
  def __init__(self):

    self.vis = meshcat.Visualizer()
    self.vis.open()
    self.vis["/Cameras/default"].set_transform(
    tf.translation_matrix([0,0,0]).dot(
    tf.euler_matrix(0,np.radians(-30),-np.pi/2)))

    self.vis["/Cameras/default/rotated/<object>"].set_transform(tf.translation_matrix([1, 0, 0]))

    self.vis["Quadrotor"].set_object(g.StlMeshGeometry.from_file('/home/alex/python/quad-sim/crazyflie2.stl'))

  def set_state(self, pos, quat):
    self.vis["Quadrotor"].set_transform(tf.translation_matrix(pos).dot(tf.quaternion_matrix(quat)))

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .exdet import ExdetDetector
from .ddd import DddDetector
from .ctdet import CtdetDetector, CtdetFeatDetector
from .multi_pose import MultiPoseDetector

detector_factory = {
  'exdet': ExdetDetector, 
  'ddd': DddDetector,
  'ctdet': CtdetDetector,
  'ctdet_feat': CtdetFeatDetector,  # ctdet which additionally outputs penultimate layer
  'multi_pose': MultiPoseDetector, 
}

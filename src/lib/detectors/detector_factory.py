from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .exdet import ExdetDetector
from .ddd import DddDetector
from .ctdet import CtdetDetector
from .multi_pose import MultiPoseDetector
from .ctdet_NFS import CtdetDetector_NFS
from .ctdet_freeze_update import CtdetDetector_NFS_freeze

detector_factory = {
  'exdet': ExdetDetector, 
  'ddd': DddDetector,
  'ctdet': CtdetDetector,
  'ctdetnfs': CtdetDetector_NFS,
  'ctdetfreeze': CtdetDetector_NFS_freeze,
  'multi_pose': MultiPoseDetector, 
}

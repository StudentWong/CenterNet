from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .exdet import ExdetDetector
from .ddd import DddDetector
from .ctdet import CtdetDetector
from .ctdet_4c import CtdetDetector as CtdetDetector4c
from .ctdet_4c_gt import CtdetDetector as CtdetDetector4c_NFS
from .ctdet_4c_adapt import CtdetDetector as CtdetDetector4c_adapt
from .multi_pose import MultiPoseDetector
from .ctdet_NFS import CtdetDetector_NFS
from .ctdet_freeze_update import CtdetDetector_NFS_freeze

detector_factory = {
  'exdet': ExdetDetector, 
  'ddd': DddDetector,
  'ctdet': CtdetDetector,
  'ctdetadaptkitti': CtdetDetector,
  'ctdetfusion': CtdetDetector4c,
  'ctdetadapt': CtdetDetector4c_adapt,
  'ctdetgtfusion': CtdetDetector4c_NFS,
  'ctdetnfs': CtdetDetector_NFS,
  'ctdetgt': CtdetDetector_NFS,
  'ctdetfreeze': CtdetDetector_NFS_freeze,
  'multi_pose': MultiPoseDetector, 
}

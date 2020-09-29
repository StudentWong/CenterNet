from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .ctdet import CtdetTrainer
from .ddd import DddTrainer
from .exdet import ExdetTrainer
from .multi_pose import MultiPoseTrainer
from .ctdet_NFS import CtdetTrainer_NFS
from .ctdet_freeze_update import CtdetTrainer_Freeze
from .ctdet_gt_centerloss import CtdetTrainer_GT_Centerloss

train_factory = {
  'exdet': ExdetTrainer, 
  'ddd': DddTrainer,
  'ctdet': CtdetTrainer,
  'ctdetnfs': CtdetTrainer_NFS,
  'ctdetfreeze': CtdetTrainer_Freeze,
  'ctdetgt': CtdetTrainer_GT_Centerloss,
  'multi_pose': MultiPoseTrainer, 
}

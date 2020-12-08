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
from .ctdet_adaptation import CtdetTrainer as CtdetTrainer_adapt
from .ctdet_adapt import CtdetTrainer as CtdetTrainer_adapt_kitti
from .ctdet_adapt_one_way import CtdetTrainer as CtdetTrainer_adapt_kitti_oneway
from .ctdet_fusion import CtdetTrainer as CtdetTrainer_fusion

train_factory = {
  'exdet': ExdetTrainer, 
  'ddd': DddTrainer,
  'ctdet': CtdetTrainer,
  'ctdetfusion': CtdetTrainer,
  'ctdetfusiondynamic': CtdetTrainer_fusion,
  'ctdetadapt': CtdetTrainer_adapt,
  'ctdetadaptkitti': CtdetTrainer_adapt_kitti,
  'ctdetadaptkittioneway': CtdetTrainer_adapt_kitti_oneway,
  'ctdetgtfusion': CtdetTrainer_GT_Centerloss,
  'ctdetnfs': CtdetTrainer_NFS,
  'ctdetfreeze': CtdetTrainer_Freeze,
  'ctdetgt': CtdetTrainer_GT_Centerloss,
  'multi_pose': MultiPoseTrainer, 
}

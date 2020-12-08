from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .sample.ddd import DddDataset
from .sample.exdet import EXDetDataset
from .sample.ctdet import CTDetDataset
from .sample.multi_pose import MultiPoseDataset
from .sample.ctdet_fusion import CTDetDataset as CTDetDatasetFusion
from .sample.ctdet_adapt import CTDetDatasetadapt

from .dataset.coco import COCO
from .dataset.pascal import PascalVOC
from .dataset.kitti import KITTI
from .dataset.coco_hp import COCOHP
from .dataset.shangqi import ShangQi
from .dataset.shangqi_3cat import ShangQi3Cat
from .dataset.FLIR import FLIR
from .dataset.FLIR_fusion import FLIR as FLIRFUSION
from .dataset.kitti_flir_adapt import KITTI_FLIR_adapt
from .dataset.FLIR_copy import FLIR as FLIR_exclude


dataset_factory = {
  'coco': COCO,
  'pascal': PascalVOC,
  'kitti': KITTI,
  'flirkitti': KITTI_FLIR_adapt,
  'coco_hp': COCOHP,
  'shangqi': ShangQi,
  'flir': FLIR,
  'flirexclude': FLIR_exclude,
  'flirfusion': FLIRFUSION,
  'shangqi3class': ShangQi3Cat,
}

_sample_factory = {
  'exdet': EXDetDataset,
  'ctdet': CTDetDataset,
  'ctdetgt': CTDetDataset,
  'ctdetadaptkitti': CTDetDatasetadapt,
  'ctdetadaptkittioneway': CTDetDatasetadapt,
  'ctdetfusion': CTDetDatasetFusion,
  'ctdetfusiondynamic': CTDetDatasetFusion,
  'ctdetadapt': CTDetDatasetFusion,
  'ctdetgtfusion': CTDetDatasetFusion,
  'ctdetnfs': CTDetDataset,
  'ctdetfreeze': CTDetDataset,
  'ddd': DddDataset,
  'multi_pose': MultiPoseDataset
}


def get_dataset(dataset, task):
  # print(dataset)
  # print(task)
  class Dataset(dataset_factory[dataset], _sample_factory[task]):
    pass
  return Dataset
  

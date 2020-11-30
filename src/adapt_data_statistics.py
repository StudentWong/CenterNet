# from src.lib.datasets.dataset.FLIR_adapt import FLIR_adapt
# from src.lib.datasets.dataset.kitti_adapt import KITTI_adapt
# from src.lib.datasets.dataset.kitti_flir_adapt import KITTI_FLIR_adapt
from lib.datasets.dataset.kitti_flir_adapt import KITTI_FLIR_adapt
from lib.datasets.sample.ctdet_adapt import CTDetDatasetadapt

class optt:
  def __init__(self):
    self.data_dir = "/home/htz/caijihuzhuo/CenterNet/data/"
    self.task = 'ctdet'
    self.keep_res = False
    self.input_h = 384
    self.input_w = 384
    self.not_rand_crop = False
    self.flip = 0.5
    self.no_color_aug = False
    self.down_ratio = 4
    self.mse_loss = False
    self.dense_wh = False
    self.cat_spec_wh = False
    self.reg_offset = True
    self.debug = 0

def get_object_list(dataset, idx):
    img_id = dataset.images[idx]
    ann_ids = dataset.coco.getAnnIds(imgIds=[img_id])
    anns = dataset.coco.loadAnns(ids=ann_ids)
    cat_list = []
    for ann in anns:
        if ann['category_id'] not in cat_list:
            cat_list = cat_list + [ann['category_id']]
    return cat_list

if __name__ == '__main__':
    #
    # opt = optt()
    # kittidata = KITTI_adapt(opt, 'train', [512, 640])
    # flirdata = FLIR_adapt(opt, 'train')
    # len_kitti = len(kittidata)
    # len_flir = len(flirdata)
    # kitti = {None: [], 1: [], 2: [], 3: []}
    # # print(kitti[None])
    # for i in range(len_kitti):
    #     obj_list = get_object_list(kittidata, i)
    #     img_id = kittidata.images[i]
    #     if len(obj_list) == 0:
    #         kitti[None] = kitti[None] + [img_id]
    #     elif 1 in obj_list:
    #         kitti[1] = kitti[1] + [img_id]
    #     elif 2 in obj_list:
    #         kitti[2] = kitti[2] + [img_id]
    #     elif 3 in obj_list:
    #         kitti[3] = kitti[3] + [img_id]
    #
    # flir = {None: [], 1: [], 2: [], 3: []}
    # for i in range(len_flir):
    #     obj_list = get_object_list(flirdata, i)
    #     img_id = flirdata.images[i]
    #     if len(obj_list) == 0:
    #         flir[None] = flir[None] + [img_id]
    #     elif 1 in obj_list:
    #         flir[1] = flir[1] + [img_id]
    #     elif 2 in obj_list:
    #         flir[2] = flir[2] + [img_id]
    #     elif 3 in obj_list:
    #         flir[3] = flir[3] + [img_id]
    # print(kitti)
    # print(flir)

    opt = optt()
    class Dataset(KITTI_FLIR_adapt, CTDetDatasetadapt):
        pass
    data = Dataset(opt, 'train')
    data[3]

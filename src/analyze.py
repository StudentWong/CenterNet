import argparse
import matplotlib.pyplot as plt
import re
import numpy as np

#python analyze.py --logpath /home/lilium/windows_disk2/caijihuzhuo/CenterNet/exp/ctdet/FLIR_origin_benchmark_exclude_True/logs_2020-11-22-15-26/log.txt --train_val val --item loss
#python analyze.py --logpath /home/lilium/windows_disk2/caijihuzhuo/CenterNet/exp/ctdet/FLIR_origin_benchmark_exclude/logs_2020-11-22-15-19/log.txt --train_val val --item loss
#python analyze.py --logpath /home/lilium/windows_disk2/caijihuzhuo/CenterNet/exp/ctdetadapt/adapt_Tweight0.8_cut0.8/logs_2020-11-21-12-41/log.txt --train_val val --item loss
#python analyze.py --logpath /home/lilium/windows_disk2/caijihuzhuo/CenterNet/exp/ctdetadapt/adapt_Tweight0.8_cut0.5/logs_2020-11-22-03-16/log.txt --train_val val --item loss


parser = argparse.ArgumentParser()
parser.add_argument('--logpath', default='',
                             help='logpath')
parser.add_argument('--train_val', default='val',
                             help='train or val')                          
parser.add_argument('--item', default='',
                             help='log item')

if __name__ == "__main__":
    opt = parser.parse_args()
    print(opt)
    log_path = opt.logpath
    item = opt.item
    assert opt.train_val=="train" or opt.train_val=="val"
    # print(log_path)
    AP_list = []
    val_loss_list = []
    with open(log_path, 'r') as fp:
        lines = fp.readlines()
        # print(lines[0])
        for line in lines:
            AP_partten = re.compile("AP50 \d+.\d+")
            AP_result_ = re.findall(AP_partten, line)
            assert len(AP_result_) == 1
            AP_result = AP_result_[0]
            AP_list = AP_list + [AP_result]
            # print(AP_result)
            item_partten_str = "\| ?" + item + " \d+.\d+"
            # print(item_partten_str)
            item_partten = re.compile(item_partten_str)
            item_result_ = re.findall(item_partten, line)
            # print(item_result_)
            assert len(item_result_) == 2
            if opt.train_val == "train":
                item_result = item_result_[0]
            elif opt.train_val == "val":
                item_result = item_result_[1]
            val_loss_list = val_loss_list + [item_result]
            # print(item_result)
    val_loss_list_np = np.array(val_loss_list)
    idx = np.argmin(val_loss_list_np)
    print(AP_list[idx])
    


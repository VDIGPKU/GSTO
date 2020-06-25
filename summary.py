# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import argparse
import torch
import torch.optim
from config import config
from config import update_config
from utils.modelsummary import get_model_summary
from models.seg_hrnet import get_seg_model


def parse_args():
    yaml_file= '/home/wangzhuoying/test-hrnet/HRNet-Semantic-Segmentation-master/experiments/cityscapes/seg_hrnet_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml'
    yaml_file1 = '/home/wangzhuoying/test-hrnet/HRNet-Semantic-Segmentation-master/experiments/cityscapes/seg_hrnet_w18_small_v2_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml'
    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default=yaml_file,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args

def main():

    args = parse_args()
    # build model
    model = get_seg_model(config)

    dump_input = torch.rand(
        (1, 3, config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    )
    print(get_model_summary(model.cuda(), dump_input.cuda()))



if __name__ == '__main__':
    main()

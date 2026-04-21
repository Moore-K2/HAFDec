from __future__ import absolute_import, division, print_function

from options import LiteMonoOptions
from options_parse_yaml import LiteMonoOptionsYaml
from trainer import Trainer

#TODO 解决windows多进程问题
import multiprocessing
multiprocessing.freeze_support()


# options = LiteMonoOptions()
# opts = options.parse()

# 用yaml配置文件训练
options = LiteMonoOptionsYaml()
opts = options.parse()


def main():

    opts.data_path = r'E:\datasets\kitti_data'
    opts.split = 'eigen_zhou'
    opts.model_name = "LiteMonov2_small_HAFDepthDecoder_p"
    opts.gpu_id = 1
    opts.mypretrain = '/home/RM_luo/Documents/Lite-Mono/pretrained/lite-mono-pretrain.pth'
    opts.depth_encoder = "LiteMonov2"
    opts.model = "lite-mono-small" # choices: [lite-mono, lite-mono-small, lite-mono-tiny, lite-mono-8m]
    opts.depth_decoder = "HAFDepthDecoder"
    opts.val_of_epochs = 12
    opts.warmup_steps = 2
    # lr: [0.0001, 5.0e-6, 31, 0.0001, 1.0e-5, 31]
    opts.lr= [1.0e-4, 5.0e-6, 31, 1.0e-4, 1.0e-5, 31]
    # opts.lr = [5.0e-4, 5.0e-6, 35, 5.0e-4, 1.0e-5, 35]
    opts.batch_size = 16
    opts.num_epochs = 31
    trainer = Trainer(opts)
    trainer.train_v1()



if __name__ == "__main__":
    main()

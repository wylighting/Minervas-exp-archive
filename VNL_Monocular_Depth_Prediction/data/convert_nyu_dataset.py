from nyudv2_dataset import NYUDV2Dataset
import cv2

from tools.parse_arg_train import TrainOptions
train_opt = TrainOptions()
train_args = train_opt.parse()
train_args.thread = 0
train_opt.print_options(train_args)

dataset = NYUDV2Dataset()
dataset.initialize(train_opt)


for i in range(len(dataset.data_size)):
    rgb_img = dataset.A[i]
    rgb_fn = dataset.A_paths[i]
    cv2.imwrite(rgb_fn, rgb_img)
    depth = dataset.B[i]
    depth_fn = dataset.B_paths[i]
    cv2.imwrite(depth_fn, depth)
    break
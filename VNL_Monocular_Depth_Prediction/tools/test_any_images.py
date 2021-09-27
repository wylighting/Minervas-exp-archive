import os
import sys
import cv2
import torch
import numpy as np
# uncomment the line below if you face "No module named 'lib'"" problem
sys.path.append(os.path.abspath('./'))
from lib.utils.net_tools import load_ckpt
from lib.utils.logging import setup_logging
import torchvision.transforms as transforms
from tools.parse_arg_test import TestOptions
from data.load_dataset import CustomerDataLoader
from lib.models.metric_depth_model import MetricDepthModel
from lib.core.config import cfg, merge_cfg_from_file
from lib.models.image_transfer import bins_to_depth
import matplotlib.pyplot as plt
from tqdm import tqdm

logger = setup_logging(__name__)


def scale_torch(img, scale):
    """
    Scale the image and output it in torch.tensor.
    :param img: input image. [C, H, W]
    :param scale: the scale factor. float
    :return: img. [C, H, W]
    """
    img = np.transpose(img, (2, 0, 1))
    img = img[::-1, :, :]
    img = img.astype(np.float32)
    img /= scale
    img = torch.from_numpy(img.copy())
    img = transforms.Normalize(cfg.DATASET.RGB_PIXEL_MEANS, cfg.DATASET.RGB_PIXEL_VARS)(img)
    return img


if __name__ == '__main__':
    test_args = TestOptions().parse()
    test_args.thread = 1
    test_args.batchsize = 1
    merge_cfg_from_file(test_args)

    data_loader = CustomerDataLoader(test_args)
    test_datasize = len(data_loader)
    logger.info('{:>15}: {:<30}'.format('test_data_size', test_datasize))
    # load model
    model = MetricDepthModel()

    model.eval()

    # load checkpoint
    if test_args.load_ckpt:
        load_ckpt(test_args, model)
    model.cuda()
    model = torch.nn.DataParallel(model)

    # path = os.path.join(cfg.ROOT_DIR, './test_any_imgs_examples') # the dir of imgs
    path = '../datasets/NYUDV2_origin/test/test_rgb'
    out_path = '../VNL_Monocular_Depth_Prediction/results/debug'
    os.makedirs(out_path, exist_ok=True)
    imgs_list = os.listdir(path)
    for i in tqdm(imgs_list):
        # print(i)
        with torch.no_grad():
            img = cv2.imread(os.path.join(path, i))
            print('original', img.shape)
            img_resize = cv2.resize(img, (int(img.shape[1]), int(img.shape[0])), interpolation=cv2.INTER_LINEAR)
            img_torch = scale_torch(img_resize, 255)
            # print(img_torch.shape)
            img_torch = img_torch[None, :, :, :].cuda()
            # print(img_torch.shape)

            _, pred_depth_softmax= model.module.depth_model(img_torch)
            pred_depth = bins_to_depth(pred_depth_softmax)
            pred_depth = pred_depth.cpu().numpy().squeeze()
            # pred_depth_scale = (pred_depth / pred_depth.max() * 60000).astype(np.uint16)  # scale 60000 for visualization

            # cv2.imwrite(os.path.join(path, i.split('.')[0] + '-raw.png'), pred_depth_scale)
            plt.imsave(os.path.join(out_path, i.split('.')[0] + '-vnl-r.png'), pred_depth, cmap='rainbow')
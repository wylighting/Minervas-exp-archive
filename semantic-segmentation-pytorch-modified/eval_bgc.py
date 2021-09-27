# System libs
import os
import time
import argparse
from distutils.version import LooseVersion
# Numerical libs
import cv2
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
# Our libs
from mit_semseg.config import cfg
from mit_semseg.dataset_bgc import ValDataset
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import AverageMeter, colorEncode, accuracy, intersectionAndUnion, setup_logger
from mit_semseg.lib.nn import user_scattered_collate, async_copy_to
from mit_semseg.lib.utils import as_numpy
from PIL import Image
from tqdm import tqdm


colors = np.load('./data/colormap.npy')[:, [2, 1, 0]]


def overlay_mask(im, masks, alpha=0.5):
    ov = im.copy()
    im = im.astype(np.float32)
    total_ma = np.zeros([im.shape[0], im.shape[1]])
    for index in np.unique(masks):
        if index == 0:
            continue
        ma = (masks == index).astype(np.bool)
        fg = im * alpha + np.ones(im.shape) * (1 - alpha) * colors[index]
        ov[ma == 1] = fg[ma == 1]
        total_ma += ma
        contours = cv2.findContours(ma.copy().astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        cv2.drawContours(ov, contours[0], -1, (0.0, 0.0, 0.0), 1)
    ov[total_ma == 0] = im[total_ma == 0]

    return ov


def visualize_result(data, pred, metrics, dir_result):
    (img, seg, info) = data

    # prediction
    im_vis = overlay_mask(img, pred)

    img_name = info.split('/')[-1]

    im_vis = Image.fromarray(im_vis.astype(np.uint8))
    im_vis = im_vis.resize((1024, 512), Image.BILINEAR)
    im_vis.save(os.path.join(dir_result, img_name))

    np.savetxt(os.path.join(dir_result, img_name.replace('.png', '.txt')), np.array([metrics]), fmt='%f')


def save_result(data, pred, metrics, dir_result):
    (img, seg, info) = data

    img_name = info.split('/')[-1]

    pred = Image.fromarray(pred.astype(np.uint8))
    pred = pred.resize((1024, 512), Image.NEAREST)
    pred.save(os.path.join(dir_result, img_name))

    np.savetxt(os.path.join(dir_result, img_name.replace('.png', '.txt')), np.array([metrics]), fmt='%f')


def evaluate(segmentation_module, loader, cfg, gpu):
    acc_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    time_meter = AverageMeter()

    segmentation_module.eval()

    pbar = tqdm(total=len(loader))
    for batch_data in loader:
        # process data
        batch_data = batch_data[0]
        seg_label = as_numpy(batch_data['seg_label'][0])
        img_resized = batch_data['img_data']

        torch.cuda.synchronize()
        tic = time.perf_counter()
        with torch.no_grad():
            segSize = (seg_label.shape[0], seg_label.shape[1])
            scores = torch.zeros(1, cfg.DATASET.num_class, segSize[0], segSize[1])
            scores = async_copy_to(scores, gpu)

            feed_dict = batch_data.copy()
            feed_dict['img_data'] = img_resized
            del feed_dict['img_ori']
            del feed_dict['info']
            feed_dict = async_copy_to(feed_dict, gpu)
            # feed_dict = list(feed_dict) #New add: for training with validation loop

            # forward pass
            scores = scores + segmentation_module(feed_dict, segSize=segSize)

            _, pred = torch.max(scores, dim=1)
            pred = as_numpy(pred.squeeze(0).cpu())

        torch.cuda.synchronize()
        time_meter.update(time.perf_counter() - tic)

        # calculate accuracy
        acc, pix = accuracy(pred, seg_label)
        intersection, union = intersectionAndUnion(pred, seg_label, cfg.DATASET.num_class)
        acc_meter.update(acc, pix)
        intersection_meter.update(intersection)
        union_meter.update(union)

        # visualization
        if cfg.VAL.visualize:
            save_result(
                (batch_data['img_ori'], seg_label, batch_data['info']),
                pred,
                acc,
                os.path.join(cfg.DIR, 'result')
            )

        pbar.update(1)

    # summary
    iou = intersection_meter.sum / (union_meter.sum + 1e-10)
    # for i, _iou in enumerate(iou):
    #     print('class [{}], IoU: {:.4f}'.format(i, _iou*100))

    print('[Eval Summary]:')
    print('Mean IoU: {:.2f}, Accuracy: {:.2f}'.format(iou.mean()*100, acc_meter.average()*100))
    return (iou.mean()*100, acc_meter.average()*100)


def main(cfg, gpu):
    torch.cuda.set_device(gpu)

    # Network Builders
    net_encoder = ModelBuilder.build_encoder(
        arch=cfg.MODEL.arch_encoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_encoder)
    net_decoder = ModelBuilder.build_decoder(
        arch=cfg.MODEL.arch_decoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        num_class=cfg.DATASET.num_class,
        weights=cfg.MODEL.weights_decoder,
        use_softmax=True)

    crit = nn.NLLLoss(ignore_index=-1)

    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)

    # Dataset and Loader
    dataset_val = ValDataset(
        cfg.DATASET.root_dataset,
        cfg.DATASET.list_val,
        cfg.DATASET)
    loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=cfg.VAL.batch_size,
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=5,
        drop_last=True)

    segmentation_module.cuda()

    # Main loop
    evaluate(segmentation_module, loader_val, cfg, gpu)

    print('Evaluation Done!')


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Validation"
    )
    parser.add_argument(
        "--cfg",
        default="config/ade20k-resnet50dilated-ppm_deepsup.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--gpu",
        default=0,
        help="gpu to use"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    # cfg.freeze()

    logger = setup_logger(distributed_rank=0)   # TODO
    logger.info("Loaded configuration file {}".format(args.cfg))
    # logger.info("Running with config:\n{}".format(cfg))

    # absolute paths of model weights
    cfg.MODEL.weights_encoder = os.path.join(
        cfg.DIR, 'encoder_' + cfg.VAL.checkpoint)
    cfg.MODEL.weights_decoder = os.path.join(
        cfg.DIR, 'decoder_' + cfg.VAL.checkpoint)
    assert os.path.exists(cfg.MODEL.weights_encoder) and \
        os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"

    if not os.path.isdir(os.path.join(cfg.DIR, "result")):
        os.makedirs(os.path.join(cfg.DIR, "result"))
    # print(args.gpu, type(args.gpu))
    # exit()

    main(cfg, args.gpu)

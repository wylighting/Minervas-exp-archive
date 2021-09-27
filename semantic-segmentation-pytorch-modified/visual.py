import os
import json
import shutil

import cv2
import numpy as np
from PIL import Image

# (0, 0, 0),
# (174, 199, 232),  # 1: wall
# (152, 223, 138),	# 2: floor
# (188, 189, 34),   # 3: chair
# (140, 86, 75),    # 4: sofa
# (214, 39, 40),    # 5: door
# (197, 176, 213),  # 6: window
# (66, 188, 102),   # 7: bookcase
# (78, 71, 183),    # 8: ceiling
# (255, 152, 150),  # 9: table


def overlay_mask(im, masks, colors, alpha=0.5):
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


def visualize_structured3d_gt(root_src="/p300/Structured3D", root_tgt='./ckpt/structured3d', size=(1024, 512)):

    list_sample = [json.loads(x.rstrip()) for x in open('./data/S3D_40_testing.odgt', 'r')]

    for this_record in list_sample:
        print(this_record['fpath_img'])
        image_path = os.path.join(root_src, this_record['fpath_img'])
        segm_path = os.path.join(root_src, this_record['fpath_segm'])

        img = Image.open(image_path).convert('RGB')
        segm = Image.open(segm_path)

        img = img.resize(size, Image.BILINEAR)
        segm = segm.resize(size, Image.NEAREST)

        img = np.array(img)
        segm = np.array(segm)

        Image.fromarray(img.astype(np.uint8)).save(os.path.join(root_tgt, 'rgb', this_record['fpath_img'].split('/')[-1]))
        Image.fromarray(segm.astype(np.uint8)).save(os.path.join(root_tgt, 'gt', this_record['fpath_img'].split('/')[-1]))


def visualize_stanford_gt(root_src="/p300/Stanford2D3D", root_tgt='./ckpt/stanford', size=(1024, 512)):

    list_sample = [json.loads(x.rstrip()) for x in open('./data/2D3DS_testing.odgt', 'r')]

    for this_record in list_sample:
        print(this_record['fpath_img'])
        image_path = os.path.join(root_src, this_record['fpath_img'])
        segm_path = os.path.join(root_src, this_record['fpath_segm'])

        img = Image.open(image_path).convert('RGB')
        segm = Image.open(segm_path)

        img = img.resize(size, Image.BILINEAR)
        segm = segm.resize(size, Image.NEAREST)

        img = np.array(img)
        segm = np.array(segm)

        Image.fromarray(img.astype(np.uint8)).save(os.path.join(root_tgt, 'rgb', this_record['fpath_img'].split('/')[-1]))
        Image.fromarray(segm.astype(np.uint8)).save(os.path.join(root_tgt, 'gt', this_record['fpath_img'].split('/')[-1]))


def select_structured3d():
    colors = np.load('data/colormap_40.npy')[:, [2, 1, 0]]

    root = os.path.join('visual', 'structured3d')

    os.makedirs(os.path.join(root, 'pspnet'), exist_ok=True)
    os.makedirs(os.path.join(root, 'upernet'), exist_ok=True)
    os.makedirs(os.path.join(root, 'hrnetv2'), exist_ok=True)

    # filelist = []
    # for filename in sorted(os.listdir(os.path.join('ckpt', 'structured3d', 'gt'))):
    #     acc_upernet = np.loadtxt(os.path.join("ckpt", "upernet-s-40", "result", filename.replace('.png', '.txt')))
    #     acc_hrnetv2 = np.loadtxt(os.path.join("ckpt", "hrnetv2-s-40", "result", filename.replace('.png', '.txt')))
    #     filelist.append([filename, acc_hrnetv2 - acc_upernet])
    # filelist = sorted(filelist, key=lambda s: s[-1], reverse=True)

    for filename in sorted(os.listdir(os.path.join('ckpt', 'structured3d', 'gt'))):
        print(filename)

        acc_pspnet = np.loadtxt(os.path.join("ckpt", "pspnet-s-40", "result", filename.replace('.png', '.txt')))
        acc_upernet = np.loadtxt(os.path.join("ckpt", "upernet-s-40", "result", filename.replace('.png', '.txt')))
        acc_hrnetv2 = np.loadtxt(os.path.join("ckpt", "hrnetv2-s-40", "result", filename.replace('.png', '.txt')))

        img = np.array(Image.open(os.path.join("ckpt", "structured3d", 'rgb', filename)))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        gt = np.array(Image.open(os.path.join("ckpt", "structured3d", "gt", filename)))
        pred_pspnet = np.array(Image.open(os.path.join("ckpt", "pspnet-s-40", "result", filename))) + 1
        pred_upernet = np.array(Image.open(os.path.join("ckpt", "upernet-s-40", "result", filename))) + 1
        pred_hrnetv2 = np.array(Image.open(os.path.join("ckpt", "hrnetv2-s-40", "result", filename))) + 1

        if np.sum(gt == 4) == 0:
            continue

        mask = (gt != 0)
        # mask = (gt != 0) * (gt != 38) * (gt != 39) * (gt != 40)

        img_gt = overlay_mask(img, gt, colors)
        img_pspnet = overlay_mask(img, pred_pspnet * mask, colors)
        img_upernet = overlay_mask(img, pred_upernet * mask, colors)
        img_hrnetv2 = overlay_mask(img, pred_hrnetv2 * mask, colors)

        print(f"pspnet: {acc_pspnet*100:.2f}\tupernet: {acc_upernet*100:.2f}\thrnetv2: {acc_hrnetv2*100:.2f}")

        img_row_1 = np.concatenate((img_pspnet, img_upernet), axis=1)
        img_row_2 = np.concatenate((img_hrnetv2, img_gt), axis=1)
        img_concat = np.concatenate((img_row_1, img_row_2))

        img_concat = cv2.resize(img_concat, (1024, 512))

        img_row_1 = np.concatenate((pred_pspnet, pred_upernet), axis=1)
        img_row_2 = np.concatenate((pred_hrnetv2, gt), axis=1)
        mask_concat = np.concatenate((img_row_1, img_row_2))

        mask_concat = cv2.resize(mask_concat, (1024, 512), interpolation=cv2.INTER_NEAREST)

        cv2.imshow('visual', img_concat)
        cv2.imshow('gt', mask_concat)

        k = cv2.waitKey(0)
        if k == ord('s'):
            img = cv2.resize(img, (512, 256))
            img_gt = cv2.resize(img_gt, (512, 256))
            img_pspnet = cv2.resize(img_pspnet, (512, 256))
            img_upernet = cv2.resize(img_upernet, (512, 256))
            img_hrnetv2 = cv2.resize(img_hrnetv2, (512, 256))

            cv2.imwrite(os.path.join(root, 'rgb', filename), img)
            cv2.imwrite(os.path.join(root, 'gt', filename), img_gt)
            cv2.imwrite(os.path.join(root, 'pspnet', filename), img_pspnet)
            cv2.imwrite(os.path.join(root, 'upernet', filename), img_upernet)
            cv2.imwrite(os.path.join(root, 'hrnetv2', filename), img_hrnetv2)

        elif k == ord('n'):  # normally -1 returned, so don't print it
            continue

        elif k == ord('q'):  # normally -1 returned, so don't print it
            break


def select_stanford():
    colors = np.load('data/colormap.npy')[:, [2, 1, 0]]

    root = os.path.join('visual', 'stanford2d3d')
    arch = 'pspnet'

    # os.makedirs(os.path.join('visual', arch, 's'), exist_ok=True)
    os.makedirs(os.path.join('visual', arch, 'r'), exist_ok=True)
    # os.makedirs(os.path.join('visual', arch, 's2r'), exist_ok=True)
    os.makedirs(os.path.join('visual', arch, 's+r'), exist_ok=True)

    filelist = []
    for filename in sorted(os.listdir('./ckpt/gt')):
        acc_swr = np.loadtxt(os.path.join("ckpt", f"{arch}-s+r", "result", filename.replace('.png', '.txt')))
        acc_r = np.loadtxt(os.path.join("ckpt", f"{arch}-r", "result", filename.replace('.png', '.txt')))
        filelist.append([filename, acc_swr - acc_r])
    filelist = sorted(filelist, key=lambda s: s[-1], reverse=True)

    for filename, _ in filelist:
        print(filename)

        # acc_s = np.loadtxt(os.path.join("ckpt", f"{arch}-s", "result", filename.replace('.png', '.txt')))
        acc_r = np.loadtxt(os.path.join("ckpt", f"{arch}-r", "result", filename.replace('.png', '.txt')))
        # acc_s2r = np.loadtxt(os.path.join("ckpt", f"{arch}-s-r", "result", filename.replace('.png', '.txt')))
        acc_swr = np.loadtxt(os.path.join("ckpt", f"{arch}-s+r", "result", filename.replace('.png', '.txt')))

        img = np.array(Image.open(os.path.join("ckpt", "rgb", filename)))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        gt = np.array(Image.open(os.path.join("ckpt", "gt", filename)))
        # pred_s = np.array(Image.open(os.path.join("ckpt", f"{arch}-s", "result", filename))) + 1
        pred_r = np.array(Image.open(os.path.join("ckpt", f"{arch}-r", "result", filename))) + 1
        # pred_s2r = np.array(Image.open(os.path.join("ckpt", f"{arch}-s-r", "result", filename))) + 1
        pred_swr = np.array(Image.open(os.path.join("ckpt", f"{arch}-s+r", "result", filename))) + 1

        mask = (gt != 0)

        img_gt = overlay_mask(img, gt, colors)
        # img_s = overlay_mask(img, pred_s * mask, colors)
        img_r = overlay_mask(img, pred_r * mask, colors)
        # img_s2r = overlay_mask(img, pred_s2r * mask, colors)
        img_swr = overlay_mask(img, pred_swr * mask, colors)

        # print(f"s: {acc_s*100:.2f}\tr: {acc_r*100:.2f}\ts2r: {acc_s2r*100:.2f}\ts+r: {acc_swr*100:.2f}")
        print(f"r: {acc_r*100:.2f}\ts+r: {acc_swr*100:.2f}")

        img_row_1 = np.concatenate((img, img_r), axis=1)
        img_row_2 = np.concatenate((img_gt, img_swr), axis=1)
        img_concat = np.concatenate((img_row_1, img_row_2))

        img_concat = cv2.resize(img_concat, (1536, 512))

        cv2.imshow('visual', img_concat)

        k = cv2.waitKey(0)
        if k == ord('s'):
            img = cv2.resize(img, (512, 256))
            img_gt = cv2.resize(img_gt, (512, 256))
            # img_s = cv2.resize(img_s, (512, 256))
            img_r = cv2.resize(img_r, (512, 256))
            # img_s2r = cv2.resize(img_s2r, (512, 256))
            img_swr = cv2.resize(img_swr, (512, 256))

            cv2.imwrite(os.path.join(root, arch, 'rgb', filename), img)
            cv2.imwrite(os.path.join(root, arch, 'gt', filename), img_gt)
            # cv2.imwrite(os.path.join(root, arch, 's', filename), img_s)
            cv2.imwrite(os.path.join(root, arch, 'r', filename), img_r)
            # cv2.imwrite(os.path.join(root, arch, 's2r', filename), img_s2r)
            cv2.imwrite(os.path.join(root, arch, 's+r', filename), img_swr)

        elif k == ord('n'):  # normally -1 returned, so don't print it
            continue
        elif k == ord('q'):  # normally -1 returned, so don't print it
            break


if __name__ == "__main__":
    # visualize_structured3d_gt()

    # visualize_stanford_gt(root_src='./dataset/Stanford2D3D')

    # select_structured3d()
    
    select_stanford()

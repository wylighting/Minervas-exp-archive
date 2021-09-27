import os
import json
import cv2
import shutil
import numpy as np

from misc.pano import draw_boundary_from_cor_id

def draw_gt(w=1024, h=512):
    data_root = './Minervas_Layout_filtered_pdf_final'
    save_root = './imgs/gt_manhanttan2'

    exp = 'gt'
    os.makedirs(os.path.join(save_root, exp), exist_ok=True)
    for idx, imagename in enumerate(os.listdir(os.path.join(data_root, 'img'))):
        imagename = ''
        print(idx, imagename)
        filename = imagename.split('.')[0]

        with open(os.path.join(data_root, 'label_cor', filename+'.txt')) as f:
            gt_cor_id = np.array([l.split() for l in f], np.float32)

        img_src = cv2.imread(os.path.join(data_root, 'img', imagename))
        # img_viz = draw_boundary_from_cor_id(gt_cor_id, img_src)
        img_viz = img_src

        img_viz = cv2.resize(img_viz, (512, 256))
        cv2.imwrite(os.path.join(save_root, exp, imagename), img_viz)


def draw(w=1024, h=512):
    # data_root = './data/layoutnet_dataset'
    data_root = './data/mp3d'
    # pred_root = './assets/cubiod_inferenced'
    # pred_root = './assets/general_inferenced'
    pred_root = './output/mp3d'
    # save_root = '/home/jia/Downloads/imgs/layoutnet'
    save_root = './imgs/final_manhattan_cmp'

    # for exp in ['r_fix', 's+r_fix']:
    for exp in ['r_ours_final2', 's+r_pdf_finetune197_75']:
    # for exp in ['depth', 'da']:
        os.makedirs(os.path.join(save_root, exp), exist_ok=True)

        for idx, imagename in enumerate(os.listdir(os.path.join(data_root, 'test', 'img'))):
            print(idx, imagename)

            filename = imagename.split('.')[0]

            with open(os.path.join(data_root, 'test', 'label_cor', filename + '.txt')) as f:
                gt_cor_id = np.array([l.split() for l in f], np.float32)

            # if 'horizonnet' in pred_root:
            with open(os.path.join(pred_root, exp, filename + '.json')) as f:
                dt = json.load(f)
            dt_cor_id = np.array(dt['uv'], np.float32)
            dt_cor_id[:, 0] *= w
            dt_cor_id[:, 1] *= h
            # else:
            #     with open(os.path.join(pred_root, exp, filename + '.txt')) as f:
            #         dt_cor_id = np.array([l.split() for l in f], np.float32)

            img_src = cv2.imread(os.path.join(data_root, 'test', 'img', imagename))
            img_viz = draw_boundary_from_cor_id(dt_cor_id, img_src, False)
            img_viz = draw_boundary_from_cor_id(gt_cor_id, img_viz)

            img_viz = cv2.resize(img_viz, (512, 256))
            cv2.imwrite(os.path.join(save_root, exp, imagename), img_viz)


if __name__ == '__main__':
    draw()
    # draw_gt()
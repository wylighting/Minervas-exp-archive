import os
import json
import torch
from torchvision import transforms
import numpy as np
from PIL import Image


def imresize(im, size, interp='bilinear'):
    if interp == 'nearest':
        resample = Image.NEAREST
    elif interp == 'bilinear':
        resample = Image.BILINEAR
    elif interp == 'bicubic':
        resample = Image.BICUBIC
    else:
        raise Exception('resample method undefined!')

    return im.resize(size, resample)


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, odgt, opt, **kwargs):
        # parse options
        self.imgMaxSize = opt.imgMaxSize
        # max down sampling rate of network to avoid rounding during conv or pooling
        self.padding_constant = opt.padding_constant

        # parse the input list
        self.list_sample = []
        self.num_sample = []
        for odgt_i in odgt.split(' '):
            list_sample, num_sample = self.parse_input_list(odgt_i, **kwargs)
            self.list_sample.append(list_sample)
            self.num_sample.append(num_sample)

        # mean and std
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def parse_input_list(self, odgt, max_sample=-1, start_idx=-1, end_idx=-1):
        if isinstance(odgt, list):
            list_sample = odgt
        elif isinstance(odgt, str):
            list_sample = [json.loads(x.rstrip()) for x in open(odgt, 'r')]

        if max_sample > 0:
            list_sample = list_sample[0:max_sample]
        if start_idx >= 0 and end_idx >= 0:     # divide file list
            list_sample = list_sample[start_idx:end_idx]

        num_sample = len(list_sample)
        assert num_sample > 0
        print('# samples: {}'.format(num_sample))
        return list_sample, num_sample

    def img_transform(self, img):
        # 0-255 to 0-1
        img = np.float32(np.array(img)) / 255.
        img = img.transpose((2, 0, 1))
        img = self.normalize(torch.from_numpy(img.copy()))
        return img

    def segm_transform(self, segm):
        # to tensor, -1 to 149
        segm = torch.from_numpy(np.array(segm)).long() - 1
        return segm

    # Round x to the nearest multiple of p and x' >= x
    def round2nearest_multiple(self, x, p):
        return ((x - 1) // p + 1) * p


class TrainDataset(BaseDataset):
    def __init__(self, root_dataset,
                 odgt, opt, batch_per_gpu=1, **kwargs):
        super(TrainDataset, self).__init__(odgt, opt, **kwargs)
        self.root_dataset = root_dataset.split(' ')
        # down sampling rate of segm labe
        self.segm_downsampling_rate = opt.segm_downsampling_rate
        self.batch_per_gpu = batch_per_gpu

        # override dataset length when trainig with batch_per_gpu > 1
        self.cur_idx = [0, 0]
        self.if_shuffled = False

    def _get_sub_batch(self):
        assert self.batch_per_gpu == 2
        batch_records = []
        for i in range(len(self.list_sample)):
            # get a sample record
            this_sample = self.list_sample[i][self.cur_idx[i]]
            batch_records.append(this_sample)

            # update current sample pointer
            self.cur_idx[i] += 1
            if self.cur_idx[i] >= self.num_sample[i]:
                self.cur_idx[i] = 0
                np.random.shuffle(self.list_sample[i])
        return batch_records

    def __getitem__(self, index):
        # NOTE: random shuffle for the first time. shuffle in __init__ is useless
        if not self.if_shuffled:
            np.random.seed(index)
            for list_sample in self.list_sample:
                np.random.shuffle(list_sample)
            self.if_shuffled = True

        # get sub-batch candidates
        batch_records = self._get_sub_batch()

        img_height, img_width = batch_records[0]['height'], batch_records[0]['width']

        this_scale = self.imgMaxSize / max(img_height, img_width)
        
        height = int(img_width * this_scale)
        width = int(img_height * this_scale)

        # Here we must pad both input image and segmentation map to size h' and w' so that p | h' and p | w'
        batch_images = torch.zeros(
            self.batch_per_gpu, 3, height, width)
        batch_segms = torch.zeros(
            self.batch_per_gpu,
            height // self.segm_downsampling_rate,
            width // self.segm_downsampling_rate).long()

        for i in range(self.batch_per_gpu):
            this_record = batch_records[i]

            # load image and label
            image_path = os.path.join(self.root_dataset[i], this_record['fpath_img'])
            segm_path = os.path.join(self.root_dataset[i], this_record['fpath_segm'])

            img = Image.open(image_path).convert('RGB')
            segm = Image.open(segm_path)
            assert(img.size[0] == segm.size[0])
            assert(img.size[1] == segm.size[1])

            # random_flip
            if np.random.choice([0, 1]):
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                segm = segm.transpose(Image.FLIP_LEFT_RIGHT)

            # note that each sample within a mini batch has different scale param
            img = imresize(img, (width, height), interp='bilinear')
            segm = imresize(segm, (width, height), interp='nearest')

            # further downsample seg label, need to avoid seg label misalignment
            segm_rounded_width = self.round2nearest_multiple(segm.size[0], self.segm_downsampling_rate)
            segm_rounded_height = self.round2nearest_multiple(segm.size[1], self.segm_downsampling_rate)
            segm_rounded = Image.new('L', (segm_rounded_width, segm_rounded_height), 0)
            segm_rounded.paste(segm, (0, 0))
            segm = imresize(
                segm_rounded,
                (segm_rounded.size[0] // self.segm_downsampling_rate, \
                 segm_rounded.size[1] // self.segm_downsampling_rate), \
                interp='nearest')

            # image transform, to torch float tensor 3xHxW
            img = self.img_transform(img)

            # segm transform, to torch long tensor HxW
            segm = self.segm_transform(segm)

            # put into batch arrays
            batch_images[i][:, :img.shape[1], :img.shape[2]] = img
            batch_segms[i][:segm.shape[0], :segm.shape[1]] = segm

        output = dict()
        output['img_data'] = batch_images
        output['seg_label'] = batch_segms
        return output

    def __len__(self):
        return int(1e10) # It's a fake length due to the trick that every loader maintains its own list
        #return self.num_sampleclass


class ValDataset(BaseDataset):
    def __init__(self, root_dataset, odgt, opt, **kwargs):
        super(ValDataset, self).__init__(odgt, opt, **kwargs)
        self.root_dataset = root_dataset.split(' ')[0]

    def __getitem__(self, index):
        this_record = self.list_sample[0][index]
        # load image and label
        image_path = os.path.join(self.root_dataset, this_record['fpath_img'])
        segm_path = os.path.join(self.root_dataset, this_record['fpath_segm'])
        img = Image.open(image_path).convert('RGB')
        segm = Image.open(segm_path)
        assert(segm.mode == "L")
        assert(img.size[0] == segm.size[0])
        assert(img.size[1] == segm.size[1])

        ori_width, ori_height = img.size

        # calculate target height and width
        scale = self.imgMaxSize / float(max(ori_height, ori_width))
        target_height, target_width = int(ori_height * scale), int(ori_width * scale)

        # to avoid rounding in network
        target_width = self.round2nearest_multiple(target_width, self.padding_constant)
        target_height = self.round2nearest_multiple(target_height, self.padding_constant)

        # resize images
        img_resized = imresize(img, (target_width, target_height), interp='bilinear')

        # image transform, to torch float tensor 3xHxW
        img_resized = self.img_transform(img_resized)
        img_resized = torch.unsqueeze(img_resized, 0)

        # segm transform, to torch long tensor HxW
        segm = self.segm_transform(segm)
        batch_segms = torch.unsqueeze(segm, 0)

        output = dict()
        output['img_ori'] = np.array(img)
        output['img_data'] = img_resized.contiguous()
        output['seg_label'] = batch_segms.contiguous()
        output['info'] = this_record['fpath_img']
        return output

    def __len__(self):
        return self.num_sample[0]


class TestDataset(BaseDataset):
    def __init__(self, odgt, opt, **kwargs):
        super(TestDataset, self).__init__(odgt, opt, **kwargs)

    def __getitem__(self, index):
        this_record = self.list_sample[index]
        # load image
        image_path = this_record['fpath_img']
        img = Image.open(image_path).convert('RGB')

        ori_width, ori_height = img.size

        # calculate target height and width
        scale = self.imgMaxSize / float(max(ori_height, ori_width))
        target_height, target_width = int(ori_height * scale), int(ori_width * scale)

        # to avoid rounding in network
        target_width = self.round2nearest_multiple(target_width, self.padding_constant)
        target_height = self.round2nearest_multiple(target_height, self.padding_constant)

        # resize images
        img_resized = imresize(img, (target_width, target_height), interp='bilinear')

        # image transform, to torch float tensor 3xHxW
        img_resized = self.img_transform(img_resized)
        img_resized = torch.unsqueeze(img_resized, 0)

        output = dict()
        output['img_ori'] = np.array(img)
        output['img_data'] = img_resized.contiguous()
        output['info'] = this_record['fpath_img']
        return output

    def __len__(self):
        return self.num_sample

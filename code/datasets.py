# -*- coding: UTF-8 -*-     

from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import torchvision.transforms as transforms
import os.path
import random
import numpy as np
from miscc.config import cfg

import torch.utils.data as data
from PIL import Image
import os
import os.path
import sys


if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_imgs(img_path, imsize, bbox=None,
             transform=None, normalize=None):
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])

    if transform is not None:
        img = transform(img)

    ret = []
    for i in range(cfg.TREE.BRANCH_NUM):
        if i < (cfg.TREE.BRANCH_NUM - 1):
            re_img = transforms.Scale(imsize[i])(img)
        else:
            re_img = img
        ret.append(normalize(re_img))

    return ret

class TextDataset(data.Dataset):
    def __init__(self, data_dir, split='train',
                 base_size=64, transform=None, target_transform=None):
        self.transform = transform
        self.norm = transforms.Compose([
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.target_transform = target_transform

        self.imsize = []
        for i in range(cfg.TREE.BRANCH_NUM):
            self.imsize.append(base_size)
            base_size = base_size * 2

        self.data = []
        self.data_dir = data_dir
        self.bbox = None
        split_dir = os.path.join(data_dir, split)

        self.filenames = self.load_filenames(split_dir)
        self.lables = self.load_lable(split_dir)


        if cfg.TRAIN.FLAG:
            self.iterator = self.prepair_training_pairs
        else:
            self.iterator = self.prepair_test_pairs

    def load_class_id(self, data_dir, total_num):
        class_id = np.arange(total_num)
        return class_id

    def load_lable(self, data_dir):
        embedding_filename = '/labels.pkl'
        with open(data_dir + embedding_filename, 'rb') as f:
            embeddings = pickle.load(f)
            embeddings = np.array(embeddings)
            # embedding_shape = [embeddings.shape[-1]]
            print('labels: ', embeddings.shape)
        return embeddings


    def load_filenames(self, data_dir):
        filepath = os.path.join(data_dir, 'filenames.txt')
        with open(filepath, 'r') as f:
            # filenames = f.readlines()
            filenames = f.read().splitlines()
            filenames = [str(j) for j in filenames]
        print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        return filenames

    def prepair_training_pairs(self, index):
        key = self.filenames[index]
        bbox = None
        data_dir = self.data_dir
        lables = self.lables[index, :]
        img_name = '%s/images/%s' % (data_dir, key)
        # print(img_name)
        imgs = get_imgs(img_name, self.imsize,
                        bbox, self.transform, normalize=self.norm)

        wrong_ix = random.randint(0, len(self.filenames) - 1)
        wrong_key = self.filenames[wrong_ix]
        wrong_bbox = None
        wrong_img_name = '%s/images/%s' % \
                         (data_dir, wrong_key)
        wrong_imgs = get_imgs(wrong_img_name, self.imsize,
                              wrong_bbox, self.transform, normalize=self.norm)


        if self.target_transform is not None:
            lables = self.target_transform(lables)

        return imgs, wrong_imgs, lables, key  # captions

    def prepair_test_pairs(self, index):
        key = self.filenames[index]
        bbox = None
        data_dir = self.data_dir

        lables = self.lables[index,:]

        # img_name = '%s/images/%s.jpg' % (data_dir, key)
        img_name = '%s/images/%s' % (data_dir, key)
        imgs = get_imgs(img_name, self.imsize,
                        bbox, self.transform, normalize=self.norm)

        if self.target_transform is not None:
            lables = self.target_transform(lables)

        return imgs, lables, key  # captions

    def __getitem__(self, index):
        return self.iterator(index)

    def __len__(self):
        return len(self.filenames)


import torch
import torchvision.transforms as transforms

import argparse
import random
import pprint
import datetime
import dateutil.tz
import time

from label_generated import *  # wzh :get label directly

from miscc.config import cfg, cfg_from_file

class CropPainter(object):
    torch.backends.cudnn.benchmark = True
    def __init__(self, cls, run,
                 single_generate, single_traits, single_image_name,
                 traits_path):
        self.cls = cls
        self.run = run

        self.single_generate = single_generate
        self.single_traits = single_traits
        self.single_image_name = single_image_name

        self.traits_path = traits_path

        if self.run == "train":
            # generate train label.pkl
            generated_train_batch_label(cls, traits_path)
        if self.run == "test":
            if self.single_generate == False:
                # generate test filename.txt, test label.pkl, ept imgs
                generated_test_batch_label(cls, traits_path)
            if self.single_generate == True:
                generated_test_sigle_label(cls, single_traits, single_image_name)

        args = self.parse_args(cls, run)

        if args.cfg_file is not None:
            cfg_from_file(args.cfg_file)
        if args.gpu_id != '-1':
            cfg.GPU_ID = args.gpu_id
        else:
            cfg.CUDA = False
        if args.data_dir != '':
            cfg.DATA_DIR = args.data_dir
        print('Using config:')
        pprint.pprint(cfg)
        # manualSeed
        if not cfg.TRAIN.FLAG:
            args.manualSeed = 100
        elif args.manualSeed is None:
            args.manualSeed = random.randint(1, 10000)
        random.seed(args.manualSeed)
        torch.manual_seed(args.manualSeed)
        if cfg.CUDA:
            torch.cuda.manual_seed_all(args.manualSeed)
        now = datetime.datetime.now(dateutil.tz.tzlocal())
        timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
        output_dir = '../output/%s_%s_%s' % \
                     (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)
        split_dir, bshuffle = 'train', True
        if not cfg.TRAIN.FLAG:
            # if cfg.DATASET_NAME == '0510wheat':  # need change wzh
            bshuffle = False
            split_dir = 'test'
        # Get data loader
        imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))
        print('imsize,', imsize)
        image_transform = transforms.Compose([
            transforms.Scale(int(imsize)),
            transforms.RandomHorizontalFlip()])

        if cfg.GAN.B_CONDITION:  # text to image task ##el-
            from datasets import TextDataset
            dataset = TextDataset(cfg.DATA_DIR, split_dir,
                                  base_size=cfg.TREE.BASE_SIZE,
                                  transform=image_transform)
        assert dataset
        num_gpu = len(cfg.GPU_ID.split(','))
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=cfg.TRAIN.BATCH_SIZE * num_gpu,
            drop_last=True, shuffle=bshuffle, num_workers=int(cfg.WORKERS))

        # Define models and go to train/evaluate
        if not cfg.GAN.B_CONDITION:
            from trainer import GANTrainer as trainer
        else:
            from trainer import condGANTrainer as trainer
        algo = trainer(output_dir, dataloader, imsize)

        start_t = time.time()
        if cfg.TRAIN.FLAG:
            algo.train()
        else:
            algo.evaluate(split_dir)

        # 删除之前创建的空图
        eptpath = os.path.join('../data', self.cls, 'images')
        for files in os.listdir(eptpath):
            if files[0:3] == 'ept':
                os.remove(os.path.join(eptpath, files))

        end_t = time.time()
        print('Total time for training/testing:', end_t - start_t)

    def parse_args(self, cls, run):
        parser = argparse.ArgumentParser(description='Train a GAN network')
        if cls == "Panicle" and run == "train":
            parser.add_argument('--cfg', dest='cfg_file',
                                help='optional config file',
                                default='cfg/Panicle_3stages.yml', type=str)
        if cls == "Rice" and run == "train":
            parser.add_argument('--cfg', dest='cfg_file',
                                help='optional config file',
                                default='cfg/Rice_3stages.yml', type=str)
        if cls == "Maize" and run == "train":
            parser.add_argument('--cfg', dest='cfg_file',
                                help='optional config file',
                                default='cfg/Maize_3stages.yml', type=str)
        if cls == "Cotton" and run == "train":
            parser.add_argument('--cfg', dest='cfg_file',
                                help='optional config file',
                                default='cfg/Cotton_3stages.yml', type=str)
        if cls == "Panicle" and run == "test":
            parser.add_argument('--cfg', dest='cfg_file',
                                help='optional config file',
                                default='cfg/eval_Panicle.yml', type=str)
        if cls == "Rice" and run == "test":
            parser.add_argument('--cfg', dest='cfg_file',
                                help='optional config file',
                                default='cfg/eval_Rice.yml', type=str)
        if cls == "Maize" and run == "test":
            parser.add_argument('--cfg', dest='cfg_file',
                                help='optional config file',
                                default='cfg/eval_Maize.yml', type=str)
        if cls == "Cotton" and run == "test":
            parser.add_argument('--cfg', dest='cfg_file',
                                help='optional config file',
                                default='cfg/eval_Cotton.yml', type=str)
        parser.add_argument('--gpu', dest='gpu_id', type=str, default='0')
        parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
        parser.add_argument('--manualSeed', type=int, help='manual seed')
        args = parser.parse_args()
        return args

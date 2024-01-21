# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import logging
import sys
import os
from collections import namedtuple
Corner = namedtuple('Corner', 'x1 y1 x2 y2')

import cv2
from pysot.datasets.image_loader import jpeg4py_loader_w_failsafe as image_loader
#from pysot.datasets.image_loader import pil_loader as image_loader
# without convertion BGR to RGB
#from pysot.datasets.image_loader import opencv_seg_loader
# with convertion BGR to RGB
#from pysot.datasets.image_loader import opencv_loader
import os
import numpy as np
from torch.utils.data import Dataset

from pysot.utils.bbox import center2corner, Center
from pysot.datasets.augmentation import Augmentation
from pysot.core.config import cfg
import matplotlib.pyplot as plt

logger = logging.getLogger("global")

# setting opencv
pyv = sys.version[0]
if pyv[0] == '3':
    cv2.ocl.setUseOpenCL(False)


class SubDataset(object):
    def __init__(self, name, root, anno, frame_range, num_use, start_idx, num_train=3):
        cur_path = os.path.dirname(os.path.realpath(__file__))
        self.name = name
        self.root = root
        self.anno = os.path.join(cur_path, anno)
        self.frame_range = frame_range
        self.num_use = num_use
        self.start_idx = start_idx
        self.num_train = num_train
        logger.info("loading " + name)
        with open(self.anno, 'r') as f:
            meta_data = json.load(f)
            meta_data = self._filter_zero(meta_data)

        for video in list(meta_data.keys()):
            for track in meta_data[video]:
                frames = meta_data[video][track]
                frames = list(map(int,
                              filter(lambda x: x.isdigit(), frames.keys())))
                frames.sort()
                meta_data[video][track]['frames'] = frames
                if len(frames) <= 0:
                    logger.warning("{}/{} has no frames".format(video, track))
                    del meta_data[video][track]

        for video in list(meta_data.keys()):
            if len(meta_data[video]) <= 0:
                logger.warning("{} has no tracks".format(video))
                del meta_data[video]

        self.labels = meta_data
        self.num = len(self.labels)
        self.num_use = self.num if self.num_use == -1 else self.num_use
        self.videos = list(meta_data.keys())
        logger.info("{} loaded".format(self.name))
        self.path_format = '{}.{}.{}.jpg'
        self.pick = self.shuffle()

    def _filter_zero(self, meta_data):
        meta_data_new = {}
        for video, tracks in meta_data.items():
            new_tracks = {}
            for trk, frames in tracks.items():
                new_frames = {}
                for frm, bbox in frames.items():
                    if not isinstance(bbox, dict):
                        if len(bbox) == 4:
                            x1, y1, x2, y2 = bbox
                            w, h = x2 - x1, y2 - y1
                        else:
                            w, h = bbox
                        if w <= 0 or h <= 0:
                            continue
                    new_frames[frm] = bbox
                if len(new_frames) > 0:
                    new_tracks[trk] = new_frames
            if len(new_tracks) > 0:
                meta_data_new[video] = new_tracks
        return meta_data_new

    def log(self):
        logger.info("{} start-index {} select [{}/{}] path_format {}".format(
            self.name, self.start_idx, self.num_use,
            self.num, self.path_format))

    def shuffle(self):
        lists = list(range(self.start_idx, self.start_idx + self.num))
        pick = []
        while len(pick) < self.num_use:
            np.random.shuffle(lists)
            pick += lists
        return pick[:self.num_use]

    def get_image_anno(self, video, track, frame):
        if len(frame) == 1:
            frame = "{:06d}".format(frame[0])
            image_path = os.path.join(self.root, video,
                                  self.path_format.format(frame, track, 'x'))
            image_anno = self.labels[video][track][frame]
            return [image_path], [image_anno]
        image_path = []
        image_anno = []
        for f in frame:
            f = "{:06d}".format(f)
            image_p = os.path.join(self.root, video,
                                   self.path_format.format(f, track, 'x'))
            image_a = self.labels[video][track][f]
            image_path.append(image_p)
            image_anno.append(image_a)
        return image_path, image_anno

    def get_positive_pair(self, index):
        video_name = self.videos[index]
        video = self.labels[video_name]
        track_list = list(video.keys())
        track = np.random.choice(track_list)
        track_info = video[track]
        frames = track_info['frames']

        template_ind = np.random.randint(0, len(frames))

        left = max(template_ind - self.frame_range, 0)
        right = min(template_ind + self.frame_range, len(frames) - 1) + 1
        # right = len(frames)
        search_inds = frames[left:right]
        try:
            search_inds = np.random.choice(search_inds, self.num_train, replace=False)
        except:
            search_inds = np.random.choice(search_inds, self.num_train, replace=True)

        template_ind = [frames[template_ind]]
        # search_frame = np.random.choice(search_inds)
        # search_frames = [frames[s] for s in search_inds]
        return self.get_image_anno(video_name, track, template_ind), \
            self.get_image_anno(video_name, track, search_inds)

    def _get_positive_pair(self, index):
        video_name = self.videos[index]
        video = self.labels[video_name]
        track_list = list(video.keys())
        track = np.random.choice(track_list)
        track_info = video[track]
        frames = track_info['frames']

        sample_idx_max = len(frames) - self.num_train

        if sample_idx_max <=1:
            template_ind = np.random.randint(0, len(frames))

            left = max(template_ind - self.frame_range, 0)
            right = min(template_ind + self.frame_range, len(frames) - 1) + 1
            # right = len(frames)
            search_inds = frames[left:right]
            try:
                search_inds = np.random.choice(search_inds, self.num_train, replace=False)
            except:
                search_inds = np.random.choice(search_inds, self.num_train)

            search_inds.sort()

            template_frame = [frames[template_ind]]
            # search_frame = np.random.choice(search_inds)
            search_frames = [frames[s] for s in search_inds]
            return self.get_image_anno(video_name, track, template_frame), \
                self.get_image_anno(video_name, track, search_frames)

        while sample_idx_max <= 1:
            track_list.remove(track)
            if len(track_list) > 0:

                track = np.random.choice(track_list)
                track_info = video[track]
                frames = track_info['frames']
                sample_idx_max = len(frames) - self.num_train
            elif len(track_list) <= 0:
                index += 1
                try:
                    video_name = self.videos[index]
                except:
                    index -= 1
                    video_name = self.videos[index]
                video = self.labels[video_name]
                # obj 00 01 02 ....
                track_list = list(video.keys())
                track = np.random.choice(track_list)
                track_info = video[track]
                frames = track_info['frames']
                sample_idx_max = len(frames) - self.num_train

        template_ind = np.random.randint(0, sample_idx_max - 1)

        #left = max(template_frame - self.frame_range, 0)
        right = len(frames)
        #search_range = frames[left:right]
        search_inds = list(range(template_ind, right))

        search_inds = np.random.choice(search_inds, self.num_train, replace=False)

        search_inds.sort()

        template_frame = [frames[template_ind]]
        # search_frame = np.random.choice(search_inds)
        search_frames = [frames[s] for s in search_inds]
        return self.get_image_anno(video_name, track, template_frame), \
            self.get_image_anno(video_name, track, search_frames)

    def get_random_target_template(self, index=-1):
        if index == -1:
            index = np.random.randint(0, self.num)
        video_name = self.videos[index]
        video = self.labels[video_name]
        track = np.random.choice(list(video.keys()))
        track_info = video[track]
        frames = track_info['frames']
        frame = [np.random.choice(frames)]
        return self.get_image_anno(video_name, track, frame)

    def get_random_target_search(self, index=-1):
        if index == -1:
            index = np.random.randint(0, self.num)
        video_name = self.videos[index]
        video = self.labels[video_name]
        track_list = list(video.keys())
        track = np.random.choice(track_list)
        track_info = video[track]
        frames = track_info['frames']

        while len(frames)-1 < self.num_train:
            # track_list is img number
            track_list.remove(track)

            if len(track_list) > 0:
                track = np.random.choice(track_list)
                track_info = video[track]
                frames = track_info['frames']
            elif len(track_list) <= 0:
                try:
                    index += 1
                    video_name = self.videos[index]
                except:
                    index -= 1
                    video_name = self.videos[index]
                video = self.labels[video_name]
                track_list = list(video.keys())
                track = np.random.choice(track_list)
                track_info = video[track]
                frames = track_info['frames']

        search_inds = list(range(0, len(frames)))
        search_inds = np.random.choice(search_inds, self.num_train, replace=False)
        search_inds.sort()
        search_frames = [frames[s] for s in search_inds]
        return self.get_image_anno(video_name, track, search_frames)

    def __len__(self):
        return self.num


class TrkDataset(Dataset):
    def __init__(self,):
        super(TrkDataset, self).__init__()

        # create sub dataset
        self.all_dataset = []
        start = 0
        self.num = 0
        for name in cfg.DATASET.NAMES:
            subdata_cfg = getattr(cfg.DATASET, name)
            sub_dataset = SubDataset(
                    name,
                    subdata_cfg.ROOT,
                    subdata_cfg.ANNO,
                    subdata_cfg.FRAME_RANGE,
                    subdata_cfg.NUM_USE,
                    start
                )
            start += sub_dataset.num
            self.num += sub_dataset.num_use

            sub_dataset.log()
            self.all_dataset.append(sub_dataset)

        # data augmentation
        self.template_aug = Augmentation(
                cfg.DATASET.TEMPLATE.SHIFT,
                cfg.DATASET.TEMPLATE.SCALE,
                cfg.DATASET.TEMPLATE.BLUR,
                cfg.DATASET.TEMPLATE.FLIP,
                cfg.DATASET.TEMPLATE.COLOR
            )
        self.search_aug = Augmentation(
                cfg.DATASET.SEARCH.SHIFT,
                cfg.DATASET.SEARCH.SCALE,
                cfg.DATASET.SEARCH.BLUR,
                cfg.DATASET.SEARCH.FLIP,
                cfg.DATASET.SEARCH.COLOR
            )
        videos_per_epoch = cfg.DATASET.VIDEOS_PER_EPOCH
        self.num = videos_per_epoch if videos_per_epoch > 0 else self.num
        self.num *= cfg.TRAIN.EPOCH
        self.pick = self.shuffle()

    def shuffle(self):
        pick = []
        m = 0
        while m < self.num:
            p = []
            for sub_dataset in self.all_dataset:
                sub_p = sub_dataset.pick
                p += sub_p
            np.random.shuffle(p)
            pick += p
            m = len(pick)
        logger.info("shuffle done!")
        logger.info("dataset length {}".format(self.num))
        return pick[:self.num]

    def _find_dataset(self, index):
        for dataset in self.all_dataset:
            if dataset.start_idx + dataset.num > index:
                return dataset, index - dataset.start_idx

    def _get_bbox(self, image, shape):
        #print(image.shape)
        imh, imw = image.shape[:2]
        if len(shape) == 4:
            w, h = shape[2]-shape[0], shape[3]-shape[1]
        else:
            w, h = shape
        context_amount = 0.5
        exemplar_size = cfg.TRAIN.EXEMPLAR_SIZE
        wc_z = w + context_amount * (w+h)
        hc_z = h + context_amount * (w+h)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = exemplar_size / s_z
        w = w*scale_z
        h = h*scale_z
        cx, cy = imw//2, imh//2
        bbox = center2corner(Center(cx, cy, w, h))
        return bbox

    def __len__(self):
        # self.num = VIDEOS_PER_EPOCH = 600000
        return self.num

    def __getitem__(self, index):

        index = self.pick[index]
        dataset, index = self._find_dataset(index)
        #print('NAME=', dataset.name)
        gray = cfg.DATASET.GRAY and cfg.DATASET.GRAY > np.random.random()

        neg = cfg.DATASET.NEG and cfg.DATASET.NEG > np.random.random()

        if neg:
            template = dataset.get_random_target_template(index)
            search = np.random.choice(self.all_dataset).get_random_target_search()
        else:
            template, search = dataset.get_positive_pair(index)

        template_image = image_loader(template[0][0])
        search_image = [image_loader(search[0][i]) for i in range(dataset.num_train)]

        if template_image is None:
            print('error image:', template[0][0])

        # get bounding box
        template_box = self._get_bbox(template_image, template[1][0])

        search_box = [self._get_bbox(search_image[i], search[1][i]) for i in range(dataset.num_train)]

        template, target_box = self.template_aug(template_image,
                                                 template_box,
                                                 cfg.TRAIN.EXEMPLAR_SIZE,
                                                 gray=gray)
        search_list = [self.search_aug(search_image[i], search_box[i], cfg.TRAIN.SEARCH_SIZE, gray=gray) for i in range(dataset.num_train)]

        search = []
        bbox = []
        for i in range(dataset.num_train):
            search.append(search_list[i][0])
            bbox.append(search_list[i][1])
        if neg:

            bbox = [Corner(0.0, 0.0, 0.0, 0.0) for i in range(dataset.num_train)]

        cls = [np.zeros((cfg.TRAIN.OUTPUT_SIZE, cfg.TRAIN.OUTPUT_SIZE), dtype=np.int64) for i in range(dataset.num_train)]

        template = template.transpose((2, 0, 1)).astype(np.float32)
        search = [search[i].transpose((2, 0, 1)).astype(np.float32) for i in range(dataset.num_train)]

        return {
            'template': template,
            'search': search,
            'label_cls': cls,
            'neg': neg,
            'bbox': [np.array([bbox[i].x1, bbox[i].y1, bbox[i].x2, bbox[i].y2]) for i in range(dataset.num_train)],
            'target_box': np.array(target_box),
            'name': dataset.name,
        }

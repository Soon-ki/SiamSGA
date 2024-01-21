# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os

import cv2
import torch
import numpy as np
import math
import sys

sys.path.append('../')
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from pysot.core.config import cfg
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain
from pysot.models.model_sga import ModelBuilder
from toolkit.datasets import DatasetFactory

from pysot.tracker.tracker_builder import SiamGATTracker


torch.set_num_threads(5)


def main(args):
    # load config
    cfg.merge_from_file(args.config)

    # Test dataset
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_root = os.path.join(cur_dir, 'test_dataset', args.dataset)

    # set hyper parameters
    params = getattr(cfg.HP_SEARCH, args.dataset)
    cfg.TRACK.LR = params[0]
    cfg.TRACK.PENALTY_K = params[1]
    cfg.TRACK.WINDOW_INFLUENCE = params[2]

    model = ModelBuilder()

    # load model
    model = load_pretrain(model, args.snapshot).cuda().eval()

    # build tracker
    tracker = SiamGATTracker(model)

    # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=True)

    model_name = args.snapshot.split('/')[-1].split('.')[-2]

    # OPE tracking
    for v_idx, video in enumerate(dataset):
        if args.video != '':
            # test one special video
            if video.name != args.video:
                continue
        #if v_idx < 24:
        #    continue
        toc = 0
        pred_bboxes = []
        track_times = []

        for idx, (img, gt_bbox) in enumerate(video):
            tic = cv2.getTickCount()

            if idx == 0:
                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                tracker.init(img, gt_bbox_)
                pred_bbox = gt_bbox_
                pred_bboxes.append(pred_bbox)
            else:
                outputs = tracker.track(img)
                pred_bbox = outputs['bbox']
                pred_bboxes.append(pred_bbox)
            toc += cv2.getTickCount() - tic
            track_times.append((cv2.getTickCount() - tic)/cv2.getTickFrequency())
            if idx == 0:
                cv2.destroyAllWindows()
            if args.vis and idx > 0:
                if not any(map(math.isnan,gt_bbox)):
                    gt_bbox = list(map(int, gt_bbox))
                    pred_bbox = list(map(int, pred_bbox))
                    cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                                  (gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3]), (0, 255, 0), 3)
                    cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
                                  (pred_bbox[0]+pred_bbox[2], pred_bbox[1]+pred_bbox[3]), (0, 255, 255), 3)
                    cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.imshow(video.name, img)
                    cv2.waitKey(1)
        toc /= cv2.getTickFrequency()

        # save results
        if 'GOT-10k' == args.dataset:
            video_path = os.path.join('results', args.dataset, model_name, video.name)
            if not os.path.isdir(video_path):
                os.makedirs(video_path)
            result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
            with open(result_path, 'w') as f:
                for x in pred_bboxes:
                    f.write(','.join([str(i) for i in x]) + '\n')
            result_path = os.path.join(video_path,
                                       '{}_time.txt'.format(video.name))
            with open(result_path, 'w') as f:
                for x in track_times:
                    f.write("{:.6f}\n".format(x))
        else:
            model_path = os.path.join('results', args.dataset, model_name)
            if not os.path.isdir(model_path):
                os.makedirs(model_path)
            result_path = os.path.join(model_path, '{}.txt'.format(video.name))
            with open(result_path, 'w') as f:
                for x in pred_bboxes:
                    f.write(','.join([str(i) for i in x])+'\n')
        print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
            v_idx+1, video.name, toc, idx / toc))


if __name__ == '__main__':
    pth_list = ['./SiamSGA.pth',]

    dataset_list = ['NFS30', 'UAV123', 'LaSOT', 'NFS240', 'OTB100']

    for pth in pth_list:
        for ds in dataset_list:
            parser = argparse.ArgumentParser(description='siamsga tracking')
            parser.add_argument('--video', default='', type=str, help='eval one special video')
            parser.add_argument('--snapshot', type=str, default=pth, help='snapshot of models to eval')
            parser.add_argument('--vis', action='store_true', default=False, help='whether visualzie result')
            parser.add_argument('--config', type=str, default='../experiments/Siam_SGA/config.yaml', help='config file')
            parser.add_argument('--dataset', type=str, default=ds, help='datasets')
            parser.add_argument("--gpu_id", default="0", type=str, help="gpu id")
            args = parser.parse_args()
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
            main(args)

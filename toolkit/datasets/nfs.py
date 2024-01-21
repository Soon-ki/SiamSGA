import json
import os
from tqdm import tqdm
from glob import glob
import numpy as np
from .dataset import Dataset
from .video import Video


class NFSVideo(Video):
    """

    """
    def __init__(self, name, root, video_dir, init_rect, img_names, gt_rect, attr, load_img=False):
        super(NFSVideo, self).__init__(name, root, video_dir, init_rect, img_names, gt_rect, attr, load_img)

    def load_tracker(self, path, tracker_names=None, store=True):
        if not tracker_names:
            tracker_names = [x.split('/')[-1] for x in glob(path) if os.path.isdir(x)]
        if isinstance(tracker_names, str):
            tracker_names = [tracker_names]
        for name in tracker_names:
            traj_file = os.path.join(path, name, self.name + '.txt')
            if os.path.exists(traj_file):
                with open(traj_file, 'r') as f:
                    pred_traj = [list(map(float, x.strip().split(','))) for x in f.readlines()]
            else:
                print("File not exists: ", traj_file)
            if store:
                self.pred_trajs[name] = pred_traj
            else:
                return pred_traj
        self.tracker_names = list(self.pred_trajs.keys())


class NFSDataset(Dataset):
    def __init__(self, name, dataset_root, load_img=False):
        super(NFSDataset, self).__init__(name, dataset_root)
        with open(os.path.join(dataset_root, name+'.json'), 'r') as f:
            meta_data = json.load(f)

        pbar = tqdm(meta_data.keys(), desc='loading'+name, ncols=100)
        self.videos = {}
        for video in pbar:
            # video : NFS -> file_name
            pbar.set_postfix_str(video)
            self.videos[video] = NFSVideo(video,
                                          dataset_root,
                                          meta_data[video]['video_dir'],
                                          meta_data[video]['init_rect'],
                                          meta_data[video]['img_names'],
                                          meta_data[video]['gt_rect'],
                                          None)

        """
        attr = []
        for x in self.videos.values():
            attr += x.attr
        attr = set(attr)
        self.attr = {}
        self.attr['ALL'] = list(self.videos.keys())
        for x in attr:
            self.attr[x] = []
        for k, v in self.videos.items():
            for attr_ in v.attr:
                self.attr[attr_].append(k)
        """

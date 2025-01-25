import os
import pickle
import math
import shutil
import numpy as np
import lmdb as lmdb
import textgrid as tg
import pandas as pd
import torch
import glob
import json
from termcolor import colored
from loguru import logger
from collections import defaultdict
from torch.utils.data import Dataset
import torch.distributed as dist
#import pyarrow
import pickle
import librosa
import smplx
import glob

from .build_vocab import Vocab
from .utils.audio_features import Wav2Vec2Model
from .data_tools import joints_list
from .utils import rotation_conversions as rc
from .utils import other_tools
import os.path as osp
import codecs as cs
from tqdm import tqdm
from os.path import join as pjoin

class CustomDataset(Dataset):
    def __init__(self,args, loader_type, augmentation=None, kwargs=None, build_cache=True):
        self.root_path = './libs/HumanML3D/HumanML3D/'
        
        amass_motion_dir = './process_h3d_amass/HumanML3D/new_joint_vecs/'
        beatx_motion_dir = './process_h3d_beatx/HumanML3D/new_joint_vecs/'
        window_size = 64       
        self.window_size = window_size
        self.data = []
        self.lengths = []
        self.pose_norm = True
        id_list = []
        self.training_speakers = list(range(1,31))
        split_file = self.root_path + f'{loader_type}.txt'
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(pjoin(amass_motion_dir, line.strip() + '.npy'))

        ## used for humanml3d
        for name in tqdm(id_list):
            try:
                motion = np.load(name)
                if motion.shape[0] < window_size:  # remove length less than 64.
                    continue
                self.lengths.append(motion.shape[0] - window_size)
                self.data.append(motion)
            except:
                # Some motion may not exist in KIT dataset
                pass

        # ## used for beatx
        split_rule = pd.read_csv('./datasets/BEAT_SMPL/beat_v2.0.0/beat_english_v2.0.0/'+"train_test_split.csv")
        self.selected_file = split_rule.loc[(split_rule['type'] == loader_type) & (split_rule['id'].str.split("_").str[0].astype(int).isin(self.training_speakers))]
        for index, file_name in tqdm(self.selected_file.iterrows(),total=len(self.selected_file)):
            f_name = file_name["id"]
            pose_file = beatx_motion_dir + "/" + f_name + '.npy'
            try:
                motion = np.load(pose_file)
                if motion.shape[0] < window_size:  # remove length less than 64.
                    continue
                self.lengths.append(motion.shape[0] - window_size)
                self.data.append(motion)
            except:
                # Some motion may not exist in KIT dataset
                pass
            


        self.cumsum = np.cumsum([0] + self.lengths)

        # use humanml3d mean and std to normalize the motion
        self.mean = np.load('./process_h3d_beatx/HumanML3D/Mean.npy')
        self.std = np.load('./process_h3d_beatx/HumanML3D/Std.npy')
        print("Total number of motions {}, snippets {}".format(len(self.data), self.cumsum[-1]))

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return self.cumsum[-1]

    def __getitem__(self, item):
        if item != 0:
            motion_id = np.searchsorted(self.cumsum, item) - 1
            idx = item - self.cumsum[motion_id] - 1
        else:
            motion_id = 0
            idx = 0
        motion = self.data[motion_id][idx:idx + self.window_size]
        "Z Normalization"
        if self.pose_norm:
            motion = (motion - self.mean) / self.std
        motion = torch.from_numpy(motion)
        return motion


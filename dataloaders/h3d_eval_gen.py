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
import random

from .build_vocab import Vocab
from .utils.audio_features import Wav2Vec2Model
from .data_tools import joints_list
from .utils import rotation_conversions as rc
from .utils import other_tools
from .utils.word_vectorizer import WordVectorizer

import codecs as cs
from tqdm import tqdm
from os.path import join as pjoin


class CustomDataset(Dataset):
    def __init__(self, args, loader_type):
        self.args = args
        self.w_vectorizer = WordVectorizer(pjoin('/mnt/fu09a/chenbohong/gdc/motion-diffusion-model', 'glove'), 'our_vab')
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = 300
        self.unit_length = 4
        min_motion_len = args.pose_length

        motion_dir = '/mnt/fu09a/chenbohong/gdc/HumanML3D/process_smplx/HumanML3D/new_joint_vecs'
        text_dir = '/mnt/fu09a/chenbohong/gdc/HumanML3D/process_smplx/texts'
        data_dict = {}
        id_list = []            
        split_file = f'/mnt/fu09a/chenbohong/gdc/HumanML3D/process_smplx/{loader_type}.txt'
        
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        # id_list = id_list[:200]

        new_name_list = []
        length_list = []
        give_up_num = 0
        all_text = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(motion_dir, name + '.npy'))
                if (len(motion)) < self.args.pose_length :
                    give_up_num = give_up_num + 1
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag
                        all_text.append(caption)
                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag*30) : int(to_tag*30)]
                                if (len(n_motion)) < min_motion_len or (len(n_motion) >= 300):
                                    continue
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'motion': n_motion,
                                                       'length': len(n_motion),
                                                       'text':[text_dict]}
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'text': text_data}
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except:
                pass

        print(give_up_num)
        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        # save all_text into a txt file
        with open(f'./{loader_type}.txt', 'w') as f:
            for item in all_text:
                f.write("%s\n" % item)
        
        if self.args.pose_norm:
            self.mean = np.load(self.args.mean_pose_path)
            self.std = np.load(self.args.std_pose_path)
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        if len(tokens) < 20:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (20 + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:20]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        "Randomly select a segment"
        m_length = self.args.pose_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]


        "Z Normalization"
        if self.args.pose_norm:
            motion = (motion - self.mean) / self.std
        
        in_audio = np.zeros([int(68266),2])    ## 这个位置是硬编码，为了让最后经过wav encoder的audio结果长度是200
        in_audio = torch.from_numpy(in_audio).float()
        
        in_word = np.zeros([m_length])
        in_word = torch.from_numpy(in_word).int()  if self.args.word_cache else torch.from_numpy(in_word).int() 
        vid = torch.from_numpy(np.ones([m_length, 1],dtype=int) * 99)
        motion = torch.from_numpy(motion).float()
        
        #print(motion)
        return {"pose":motion, "audio":in_audio, "word":in_word, "id":vid,"prompt_text":caption,"length":m_length,
                
                "word_embeddings":word_embeddings,
                "pos_one_hots":pos_one_hots,
            "caption":caption,
            "sent_len":sent_len,
            "motion":sent_len, "m_length":m_length,
            "tokens":'_'.join(tokens)
                
                }





class CompMDMGeneratedDataset(Dataset):

    def __init__(self, generated_motion):

        self.generated_motion = generated_motion
        self.w_vectorizer = WordVectorizer(pjoin('/mnt/fu09a/chenbohong/gdc/motion-diffusion-model', 'glove'), 'our_vab')
        self.mean = np.load('/mnt/fu09a/chenbohong/gdc/momask-codes/dataset/BEAT_HumanML3D/Mean.npy')
        self.std = np.load('/mnt/fu09a/chenbohong/gdc/momask-codes/dataset/BEAT_HumanML3D/Std.npy')
        # self.mean_for_eval = np.load('/mnt/fu09a/chenbohong/gdc/motion-diffusion-model/t2m/Decomp_SP001_SM001_H512/meta/mean.npy')
        # self.std_for_eval = np.load('/mnt/fu09a/chenbohong/gdc/motion-diffusion-model/t2m/Decomp_SP001_SM001_H512/meta/std.npy')
        self.mean_for_eval = np.load('/mnt/fu09a/chenbohong/gdc/HumanML3D/process_smplx/HumanML3D/Mean.npy')
        self.std_for_eval = np.load('/mnt/fu09a/chenbohong/gdc/HumanML3D/process_smplx/HumanML3D/Std.npy')
    def __len__(self):
        return len(self.generated_motion)


    def __getitem__(self, item):
        data = self.generated_motion[item]
        motion, m_length, caption, tokens = data['motion'], data['length'], data['caption'], data['tokens']
        sent_len = data['cap_len']

        normed_motion = motion
        denormed_motion = normed_motion * self.std + self.mean
        renormed_motion = (denormed_motion - self.mean_for_eval) / self.std_for_eval  # according to T2M norms
        motion = renormed_motion
            # This step is needed because T2M evaluators expect their norm convention
        #我把这注释的原因是因为我自己所有的数据的mean和std都是一样的

        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        m_length = 240
        motion = motion[:m_length]
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens)
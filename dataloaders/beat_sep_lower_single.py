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

from .build_vocab import Vocab
from .utils.audio_features import Wav2Vec2Model
from .data_tools import joints_list
from .utils import rotation_conversions as rc
from .utils import other_tools

class CustomDataset(Dataset):
    def __init__(self, args, loader_type, augmentation=None, kwargs=None, build_cache=True):
        
        self.audio_file_path = args.audio_file_path
        self.textgrid_file_path = args.textgrid_file_path
        self.default_pose_file = "./demo/examples/2_scott_0_1_1.npz"
        
        self.args = args
        self.loader_type = loader_type

        self.rank = 0
        self.ori_stride = self.args.stride
        self.ori_length = self.args.pose_length
        self.alignment = [0,0] # for trinity
        
        self.ori_joint_list = joints_list[self.args.ori_joints]
        self.tar_joint_list = joints_list[self.args.tar_joints]
        if 'smplx' in self.args.pose_rep:
            self.joint_mask = np.zeros(len(list(self.ori_joint_list.keys()))*3)
            self.joints = len(list(self.tar_joint_list.keys()))  
            for joint_name in self.tar_joint_list:
                self.joint_mask[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1
        else:
            self.joints = len(list(self.ori_joint_list.keys()))+1
            self.joint_mask = np.zeros(self.joints*3)
            for joint_name in self.tar_joint_list:
                if joint_name == "Hips":
                    self.joint_mask[3:6] = 1
                else:
                    self.joint_mask[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1
        # select trainable joints
        self.smplx = smplx.create(
            self.args.data_path_1+"smplx_models/", 
            model_type='smplx',
            gender='NEUTRAL_2020', 
            use_face_contour=False,
            num_betas=300,
            num_expression_coeffs=100, 
            ext='npz',
            use_pca=False,
        ).cuda().eval()

        split_rule = pd.read_csv(args.data_path+"train_test_split.csv")
        self.selected_file = split_rule.loc[(split_rule['type'] == loader_type) & (split_rule['id'].str.split("_").str[0].astype(int).isin(self.args.training_speakers))]
        if args.additional_data and loader_type == 'train':
            split_b = split_rule.loc[(split_rule['type'] == 'additional') & (split_rule['id'].str.split("_").str[0].astype(int).isin(self.args.training_speakers))]
            #self.selected_file = split_rule.loc[(split_rule['type'] == 'additional') & (split_rule['id'].str.split("_").str[0].astype(int).isin(self.args.training_speakers))]
            self.selected_file = pd.concat([self.selected_file, split_b])
        if self.selected_file.empty:
            logger.warning(f"{loader_type} is empty for speaker {self.args.training_speakers}, use train set 0-8 instead")
            self.selected_file = split_rule.loc[(split_rule['type'] == 'train') & (split_rule['id'].str.split("_").str[0].astype(int).isin(self.args.training_speakers))]
            self.selected_file = self.selected_file.iloc[0:8]
        self.data_dir = args.data_path 
        
        if loader_type == "test": 
            self.args.multi_length_training = [1.0]
        self.max_length = int(args.pose_length * self.args.multi_length_training[-1])
        self.max_audio_pre_len = math.floor(args.pose_length / args.pose_fps * self.args.audio_sr)
        if self.max_audio_pre_len > self.args.test_length*self.args.audio_sr: 
            self.max_audio_pre_len = self.args.test_length*self.args.audio_sr
        
        if args.word_rep is not None:
            with open(f"{args.data_path}weights/vocab.pkl", 'rb') as f:
                self.lang_model = pickle.load(f)
                
        preloaded_dir = self.args.tmp_dir+'/' + loader_type + f"/{args.pose_rep}_cache"      

        if self.args.beat_align:
            if not os.path.exists(args.data_path+f"weights/mean_vel_{args.pose_rep}.npy"):
                self.calculate_mean_velocity(args.data_path+f"weights/mean_vel_{args.pose_rep}.npy")
            self.avg_vel = np.load(args.data_path+f"weights/mean_vel_{args.pose_rep}.npy")
            
        if build_cache and self.rank == 0:
            self.build_cache(preloaded_dir)
        self.lmdb_env = lmdb.open(preloaded_dir, readonly=True, lock=False)
        with self.lmdb_env.begin() as txn:
            self.n_samples = txn.stat()["entries"]
        
        

    
    def calculate_mean_velocity(self, save_path):
        self.smplx = smplx.create(
            self.args.data_path_1+"smplx_models/", 
            model_type='smplx',
            gender='NEUTRAL_2020', 
            use_face_contour=False,
            num_betas=300,
            num_expression_coeffs=100, 
            ext='npz',
            use_pca=False,
        ).cuda().eval()
        dir_p = self.data_dir + self.args.pose_rep + "/"
        all_list = []
        from tqdm import tqdm
        for tar in tqdm(os.listdir(dir_p)):
            if tar.endswith(".npz"):
                m_data = np.load(dir_p+tar, allow_pickle=True)
                betas, poses, trans, exps = m_data["betas"], m_data["poses"], m_data["trans"], m_data["expressions"]
                n, c = poses.shape[0], poses.shape[1]
                betas = betas.reshape(1, 300)
                betas = np.tile(betas, (n, 1))
                betas = torch.from_numpy(betas).cuda().float()
                poses = torch.from_numpy(poses.reshape(n, c)).cuda().float()
                exps = torch.from_numpy(exps.reshape(n, 100)).cuda().float()
                trans = torch.from_numpy(trans.reshape(n, 3)).cuda().float()
                max_length = 128
                s, r = n//max_length, n%max_length
                #print(n, s, r)
                all_tensor = []
                for i in range(s):
                    with torch.no_grad():
                        joints = self.smplx(
                            betas=betas[i*max_length:(i+1)*max_length], 
                            transl=trans[i*max_length:(i+1)*max_length], 
                            expression=exps[i*max_length:(i+1)*max_length], 
                            jaw_pose=poses[i*max_length:(i+1)*max_length, 66:69], 
                            global_orient=poses[i*max_length:(i+1)*max_length,:3], 
                            body_pose=poses[i*max_length:(i+1)*max_length,3:21*3+3], 
                            left_hand_pose=poses[i*max_length:(i+1)*max_length,25*3:40*3], 
                            right_hand_pose=poses[i*max_length:(i+1)*max_length,40*3:55*3], 
                            return_verts=True,
                            return_joints=True,
                            leye_pose=poses[i*max_length:(i+1)*max_length, 69:72], 
                            reye_pose=poses[i*max_length:(i+1)*max_length, 72:75],
                        )['joints'][:, :55, :].reshape(max_length, 55*3)
                    all_tensor.append(joints)
                if r != 0:
                    with torch.no_grad():
                        joints = self.smplx(
                            betas=betas[s*max_length:s*max_length+r], 
                            transl=trans[s*max_length:s*max_length+r], 
                            expression=exps[s*max_length:s*max_length+r], 
                            jaw_pose=poses[s*max_length:s*max_length+r, 66:69], 
                            global_orient=poses[s*max_length:s*max_length+r,:3], 
                            body_pose=poses[s*max_length:s*max_length+r,3:21*3+3], 
                            left_hand_pose=poses[s*max_length:s*max_length+r,25*3:40*3], 
                            right_hand_pose=poses[s*max_length:s*max_length+r,40*3:55*3], 
                            return_verts=True,
                            return_joints=True,
                            leye_pose=poses[s*max_length:s*max_length+r, 69:72], 
                            reye_pose=poses[s*max_length:s*max_length+r, 72:75],
                        )['joints'][:, :55, :].reshape(r, 55*3)
                    all_tensor.append(joints)
                joints = torch.cat(all_tensor, axis=0)
                joints = joints.permute(1, 0)
                dt = 1/30
            # first steps is forward diff (t+1 - t) / dt
                init_vel = (joints[:, 1:2] - joints[:, :1]) / dt
                # middle steps are second order (t+1 - t-1) / 2dt
                middle_vel = (joints[:, 2:] - joints[:, 0:-2]) / (2 * dt)
                # last step is backward diff (t - t-1) / dt
                final_vel = (joints[:, -1:] - joints[:, -2:-1]) / dt
                #print(joints.shape, init_vel.shape, middle_vel.shape, final_vel.shape)
                vel_seq = torch.cat([init_vel, middle_vel, final_vel], dim=1).permute(1, 0).reshape(n, 55, 3)
                #print(vel_seq.shape)
                #.permute(1, 0).reshape(n, 55, 3)
                vel_seq_np = vel_seq.cpu().numpy()
                vel_joints_np = np.linalg.norm(vel_seq_np, axis=2) # n * 55
                all_list.append(vel_joints_np)
        avg_vel = np.mean(np.concatenate(all_list, axis=0),axis=0) # 55
        np.save(save_path, avg_vel)
        
    
    def build_cache(self, preloaded_dir):
        logger.info(f"Audio bit rate: {self.args.audio_fps}")
        logger.info("Reading data '{}'...".format(self.data_dir))
        logger.info("Creating the dataset cache...")
        if self.args.new_cache:
            if os.path.exists(preloaded_dir):
                shutil.rmtree(preloaded_dir)
        if os.path.exists(preloaded_dir):
            logger.info("Found the cache {}".format(preloaded_dir))
        elif self.loader_type == "test":
            self.cache_generation(
                preloaded_dir, True, 
                0, 0,
                is_test=True)
        else:
            self.cache_generation(
                preloaded_dir, self.args.disable_filtering, 
                self.args.clean_first_seconds, self.args.clean_final_seconds,
                is_test=False)
        
    def __len__(self):
        return self.n_samples
    

    def cache_generation(self, out_lmdb_dir, disable_filtering, clean_first_seconds,  clean_final_seconds, is_test=False):
        # if "wav2vec2" in self.args.audio_rep:
        #     self.wav2vec_model = Wav2Vec2Model.from_pretrained(f"{self.args.data_path_1}/hub/transformer/wav2vec2-base-960h")
        #     self.wav2vec_model.feature_extractor._freeze_parameters()
        #     self.wav2vec_model = self.wav2vec_model.cuda()
        #     self.wav2vec_model.eval()
        
        self.n_out_samples = 0
        # create db for samples
        if not os.path.exists(out_lmdb_dir): os.makedirs(out_lmdb_dir)
        dst_lmdb_env = lmdb.open(out_lmdb_dir, map_size= int(1024 ** 3 * 500))# 500G
        n_filtered_out = defaultdict(int)
    

        #f_name = file_name["id"]
        ext = ".npz" if "smplx" in self.args.pose_rep else ".bvh"
        pose_file = self.default_pose_file
        pose_each_file = []
        trans_each_file = []
        trans_v_each_file = []
        shape_each_file = []
        audio_each_file = []
        facial_each_file = []
        word_each_file = []
        emo_each_file = []
        sem_each_file = []
        vid_each_file = []
        id_pose = "tmp" #1_wayne_0_1_1
        
        logger.info(colored(f"# ---- Building cache for Pose   {id_pose} ---- #", "blue"))
        if "smplx" in self.args.pose_rep:
            pose_data = np.load(pose_file, allow_pickle=True)
            assert 30%self.args.pose_fps == 0, 'pose_fps should be an aliquot part of 30'
            stride = int(30/self.args.pose_fps)
            pose_each_file = pose_data["poses"][::stride] 
            trans_each_file = pose_data["trans"][::stride]
            trans_each_file[:,0] = trans_each_file[:,0] - trans_each_file[0,0]
            trans_each_file[:,2] = trans_each_file[:,2] - trans_each_file[0,2]
            trans_v_each_file = np.zeros_like(trans_each_file)
            trans_v_each_file[1:,0] = trans_each_file[1:,0] - trans_each_file[:-1,0]
            trans_v_each_file[0,0] = trans_v_each_file[1,0]
            trans_v_each_file[1:,2] = trans_each_file[1:,2] - trans_each_file[:-1,2]
            trans_v_each_file[0,2] = trans_v_each_file[1,2]
            trans_v_each_file[:,1] = trans_each_file[:,1]
            shape_each_file = np.repeat(pose_data["betas"].reshape(1, 300), pose_each_file.shape[0], axis=0)
            
            assert self.args.pose_fps == 30, "should 30"
            m_data = np.load(pose_file, allow_pickle=True)
            betas, poses, trans, exps = m_data["betas"], m_data["poses"], m_data["trans"], m_data["expressions"]
            n, c = poses.shape[0], poses.shape[1]
            betas = betas.reshape(1, 300)
            betas = np.tile(betas, (n, 1))
            betas = torch.from_numpy(betas).cuda().float()
            poses = torch.from_numpy(poses.reshape(n, c)).cuda().float()
            exps = torch.from_numpy(exps.reshape(n, 100)).cuda().float()
            trans = torch.from_numpy(trans.reshape(n, 3)).cuda().float()
            max_length = 128    # 为什么这里需要一个max_length
            s, r = n//max_length, n%max_length
            #print(n, s, r)
            all_tensor = []
            for i in range(s):
                with torch.no_grad():
                    joints = self.smplx(
                        betas=betas[i*max_length:(i+1)*max_length], 
                        transl=trans[i*max_length:(i+1)*max_length], 
                        expression=exps[i*max_length:(i+1)*max_length], 
                        jaw_pose=poses[i*max_length:(i+1)*max_length, 66:69], 
                        global_orient=poses[i*max_length:(i+1)*max_length,:3], 
                        body_pose=poses[i*max_length:(i+1)*max_length,3:21*3+3], 
                        left_hand_pose=poses[i*max_length:(i+1)*max_length,25*3:40*3], 
                        right_hand_pose=poses[i*max_length:(i+1)*max_length,40*3:55*3], 
                        return_verts=True,
                        return_joints=True,
                        leye_pose=poses[i*max_length:(i+1)*max_length, 69:72], 
                        reye_pose=poses[i*max_length:(i+1)*max_length, 72:75],
                    )['joints'][:, (7,8,10,11), :].reshape(max_length, 4, 3).cpu()
                all_tensor.append(joints)
            if r != 0:
                with torch.no_grad():
                    joints = self.smplx(
                        betas=betas[s*max_length:s*max_length+r], 
                        transl=trans[s*max_length:s*max_length+r], 
                        expression=exps[s*max_length:s*max_length+r], 
                        jaw_pose=poses[s*max_length:s*max_length+r, 66:69], 
                        global_orient=poses[s*max_length:s*max_length+r,:3], 
                        body_pose=poses[s*max_length:s*max_length+r,3:21*3+3], 
                        left_hand_pose=poses[s*max_length:s*max_length+r,25*3:40*3], 
                        right_hand_pose=poses[s*max_length:s*max_length+r,40*3:55*3], 
                        return_verts=True,
                        return_joints=True,
                        leye_pose=poses[s*max_length:s*max_length+r, 69:72], 
                        reye_pose=poses[s*max_length:s*max_length+r, 72:75],
                    )['joints'][:, (7,8,10,11), :].reshape(r, 4, 3).cpu()
                all_tensor.append(joints)
            joints = torch.cat(all_tensor, axis=0) # all, 4, 3
            # print(joints.shape)
            feetv = torch.zeros(joints.shape[1], joints.shape[0])
            joints = joints.permute(1, 0, 2)
            #print(joints.shape, feetv.shape)
            feetv[:, :-1] = (joints[:, 1:] - joints[:, :-1]).norm(dim=-1)
            #print(feetv.shape)
            contacts = (feetv < 0.01).numpy().astype(float)
            # print(contacts.shape, contacts)
            contacts = contacts.transpose(1, 0)
            pose_each_file = pose_each_file * self.joint_mask
            pose_each_file = pose_each_file[:, self.joint_mask.astype(bool)]
            pose_each_file = np.concatenate([pose_each_file, contacts], axis=1)
            # print(pose_each_file.shape)
            
            
            if self.args.facial_rep is not None:
                logger.info(f"# ---- Building cache for Facial {id_pose} and Pose {id_pose} ---- #")
                facial_each_file = pose_data["expressions"][::stride]
                if self.args.facial_norm: 
                    facial_each_file = (facial_each_file - self.mean_facial) / self.std_facial
                    
        if self.args.id_rep is not None:
            vid_each_file = np.repeat(np.array(int(999)-1).reshape(1, 1), pose_each_file.shape[0], axis=0)
    
        if self.args.audio_rep is not None:
            logger.info(f"# ---- Building cache for Audio  {id_pose} and Pose {id_pose} ---- #")
            audio_file = self.audio_file_path
            if not os.path.exists(audio_file):
                logger.warning(f"# ---- file not found for Audio  {id_pose}, skip all files with the same id ---- #")
                self.selected_file = self.selected_file.drop(self.selected_file[self.selected_file['id'] == id_pose].index)

            audio_save_path = audio_file.replace("wave16k", "onset_amplitude").replace(".wav", ".npy")

            if self.args.audio_rep == "onset+amplitude":
                audio_each_file, sr = librosa.load(audio_file)
                audio_each_file = librosa.resample(audio_each_file, orig_sr=sr, target_sr=self.args.audio_sr)
                from numpy.lib import stride_tricks
                frame_length = 1024
                # hop_length = 512
                shape = (audio_each_file.shape[-1] - frame_length + 1, frame_length)
                strides = (audio_each_file.strides[-1], audio_each_file.strides[-1])
                rolling_view = stride_tricks.as_strided(audio_each_file, shape=shape, strides=strides)
                amplitude_envelope = np.max(np.abs(rolling_view), axis=1)
                # pad the last frame_length-1 samples
                amplitude_envelope = np.pad(amplitude_envelope, (0, frame_length-1), mode='constant', constant_values=amplitude_envelope[-1])
                audio_onset_f = librosa.onset.onset_detect(y=audio_each_file, sr=self.args.audio_sr, units='frames')
                onset_array = np.zeros(len(audio_each_file), dtype=float)
                onset_array[audio_onset_f] = 1.0
                # print(amplitude_envelope.shape, audio_each_file.shape, onset_array.shape)
                audio_each_file = np.concatenate([amplitude_envelope.reshape(-1, 1), onset_array.reshape(-1, 1)], axis=1)

                
            elif self.args.audio_rep == "mfcc":
                audio_each_file = librosa.feature.melspectrogram(y=audio_each_file, sr=self.args.audio_sr, n_mels=128, hop_length=int(self.args.audio_sr/self.args.audio_fps))
                audio_each_file = audio_each_file.transpose(1, 0)
                # print(audio_each_file.shape, pose_each_file.shape)
            if self.args.audio_norm and self.args.audio_rep == "wave16k": 
                audio_each_file = (audio_each_file - self.mean_audio) / self.std_audio
                
        time_offset = 0
        if self.args.word_rep is not None:
            logger.info(f"# ---- Building cache for Word   {id_pose} and Pose {id_pose} ---- #")
            word_file = self.textgrid_file_path
            if not os.path.exists(word_file):
                logger.warning(f"# ---- file not found for Word   {id_pose}, skip all files with the same id ---- #")
                self.selected_file = self.selected_file.drop(self.selected_file[self.selected_file['id'] == id_pose].index)
            word_save_path = f"{self.data_dir}{self.args.t_pre_encoder}/{id_pose}.npy"

            tgrid = tg.TextGrid.fromFile(word_file)

            for i in range(pose_each_file.shape[0]):
                found_flag = False
                current_time = i/self.args.pose_fps + time_offset
                j_last = 0
                for j, word in enumerate(tgrid[0]): 
                    word_n, word_s, word_e = word.mark, word.minTime, word.maxTime
                    if word_s<=current_time and current_time<=word_e:
                        if word_n == " ":
                            word_each_file.append(self.lang_model.PAD_token)
                        else:
                            word_each_file.append(self.lang_model.get_word_index(word_n))
                        found_flag = True
                        j_last = j
                        break
                    else: continue   
                if not found_flag: 
                    word_each_file.append(self.lang_model.UNK_token)
            word_each_file = np.array(word_each_file)


            
        if self.args.emo_rep is not None:
            logger.info(f"# ---- Building cache for Emo    {id_pose} and Pose {id_pose} ---- #")
            rtype, start = int(id_pose.split('_')[3]), int(id_pose.split('_')[3])
            if rtype == 0 or rtype == 2 or rtype == 4 or rtype == 6:
                if start >= 1 and start <= 64:
                    score = 0
                elif start >= 65 and start <= 72:
                    score = 1
                elif start >= 73 and start <= 80:
                    score = 2
                elif start >= 81 and start <= 86:
                    score = 3
                elif start >= 87 and start <= 94:
                    score = 4
                elif start >= 95 and start <= 102:
                    score = 5
                elif start >= 103 and start <= 110:
                    score = 6
                elif start >= 111 and start <= 118:
                    score = 7
                else: pass
            else:
                # you may denote as unknown in the future
                score = 0
            emo_each_file = np.repeat(np.array(score).reshape(1, 1), pose_each_file.shape[0], axis=0)    
            #print(emo_each_file)
            
        if self.args.sem_rep is not None:
            logger.info(f"# ---- Building cache for Sem    {id_pose} and Pose {id_pose} ---- #")
            sem_file = f"{self.data_dir}{self.args.sem_rep}/{id_pose}.txt" 
            sem_all = pd.read_csv(sem_file, 
                sep='\t', 
                names=["name", "start_time", "end_time", "duration", "score", "keywords"])
            # we adopt motion-level semantic score here. 
            for i in range(pose_each_file.shape[0]):
                found_flag = False
                for j, (start, end, score) in enumerate(zip(sem_all['start_time'],sem_all['end_time'], sem_all['score'])):
                    current_time = i/self.args.pose_fps + time_offset
                    if start<=current_time and current_time<=end: 
                        sem_each_file.append(score)
                        found_flag=True
                        break
                    else: continue 
                if not found_flag: sem_each_file.append(0.)
            sem_each_file = np.array(sem_each_file)
            #print(sem_each_file)
        
        filtered_result = self._sample_from_clip(
            dst_lmdb_env,
            audio_each_file, pose_each_file, trans_each_file, trans_v_each_file,shape_each_file, facial_each_file, word_each_file,
            vid_each_file, emo_each_file, sem_each_file,
            disable_filtering, clean_first_seconds, clean_final_seconds, is_test,
            ) 
        for type in filtered_result.keys():
            n_filtered_out[type] += filtered_result[type]
                            
        
        
        
#### ---------for_end------------ ####
        with dst_lmdb_env.begin() as txn:
            logger.info(colored(f"no. of samples: {txn.stat()['entries']}", "cyan"))
            n_total_filtered = 0
            for type, n_filtered in n_filtered_out.items():
                logger.info("{}: {}".format(type, n_filtered))
                n_total_filtered += n_filtered
            logger.info(colored("no. of excluded samples: {} ({:.1f}%)".format(
                n_total_filtered, 100 * n_total_filtered / (txn.stat()["entries"] + n_total_filtered)), "cyan"))
        dst_lmdb_env.sync()
        dst_lmdb_env.close()
    
    def _sample_from_clip(
        self, dst_lmdb_env, audio_each_file, pose_each_file, trans_each_file, trans_v_each_file,shape_each_file, facial_each_file, word_each_file,
        vid_each_file, emo_each_file, sem_each_file,
        disable_filtering, clean_first_seconds, clean_final_seconds, is_test,
        ):
        """
        for data cleaning, we ignore the data for first and final n s
        for test, we return all data 
        """
        # audio_start = int(self.alignment[0] * self.args.audio_fps)
        # pose_start = int(self.alignment[1] * self.args.pose_fps)
        #logger.info(f"before: {audio_each_file.shape} {pose_each_file.shape}")
        # audio_each_file = audio_each_file[audio_start:]
        # pose_each_file = pose_each_file[pose_start:]
        # trans_each_file = 
        #logger.info(f"after alignment: {audio_each_file.shape} {pose_each_file.shape}")
        #print(pose_each_file.shape)
        round_seconds_skeleton = pose_each_file.shape[0] // self.args.pose_fps  # assume 1500 frames / 15 fps = 100 s
        #print(round_seconds_skeleton)
        if audio_each_file is not None:
            if self.args.audio_rep != "wave16k":
                round_seconds_audio = len(audio_each_file) // self.args.audio_fps # assume 16,000,00 / 16,000 = 100 s
            elif self.args.audio_rep == "mfcc":
                round_seconds_audio = audio_each_file.shape[0] // self.args.audio_fps
            else:
                round_seconds_audio = audio_each_file.shape[0] // self.args.audio_sr
            if facial_each_file is not None:
                round_seconds_facial = facial_each_file.shape[0] // self.args.pose_fps
                logger.info(f"audio: {round_seconds_audio}s, pose: {round_seconds_skeleton}s, facial: {round_seconds_facial}s")
                round_seconds_skeleton = min(round_seconds_audio, round_seconds_skeleton, round_seconds_facial)
                max_round = max(round_seconds_audio, round_seconds_skeleton, round_seconds_facial)
                if round_seconds_skeleton != max_round: 
                    logger.warning(f"reduce to {round_seconds_skeleton}s, ignore {max_round-round_seconds_skeleton}s")  
            else:
                logger.info(f"pose: {round_seconds_skeleton}s, audio: {round_seconds_audio}s")
                round_seconds_skeleton = min(round_seconds_audio, round_seconds_skeleton)
                max_round = max(round_seconds_audio, round_seconds_skeleton)
                if round_seconds_skeleton != max_round: 
                    logger.warning(f"reduce to {round_seconds_skeleton}s, ignore {max_round-round_seconds_skeleton}s")
        
        clip_s_t, clip_e_t = clean_first_seconds, round_seconds_skeleton - clean_final_seconds # assume [10, 90]s
        clip_s_f_audio, clip_e_f_audio = self.args.audio_fps * clip_s_t, clip_e_t * self.args.audio_fps # [160,000,90*160,000]
        clip_s_f_pose, clip_e_f_pose = clip_s_t * self.args.pose_fps, clip_e_t * self.args.pose_fps # [150,90*15]
        
        
        for ratio in self.args.multi_length_training:
            if is_test:# stride = length for test
                cut_length = clip_e_f_pose - clip_s_f_pose
                self.args.stride = cut_length
                self.max_length = cut_length
            else:
                self.args.stride = int(ratio*self.ori_stride)
                cut_length = int(self.ori_length*ratio)
                
            num_subdivision = math.floor((clip_e_f_pose - clip_s_f_pose - cut_length) / self.args.stride) + 1
            logger.info(f"pose from frame {clip_s_f_pose} to {clip_e_f_pose}, length {cut_length}")
            logger.info(f"{num_subdivision} clips is expected with stride {self.args.stride}")
            
            if audio_each_file is not None:
                audio_short_length = math.floor(cut_length / self.args.pose_fps * self.args.audio_fps)
                """
                for audio sr = 16000, fps = 15, pose_length = 34, 
                audio short length = 36266.7 -> 36266
                this error is fine.
                """
                logger.info(f"audio from frame {clip_s_f_audio} to {clip_e_f_audio}, length {audio_short_length}")
             
            n_filtered_out = defaultdict(int)
            sample_pose_list = []
            sample_audio_list = []
            sample_facial_list = []
            sample_shape_list = []
            sample_word_list = []
            sample_emo_list = []
            sample_sem_list = []
            sample_vid_list = []
            sample_trans_list = []
            sample_trans_v_list = []
           
            for i in range(num_subdivision): # cut into around 2s chip, (self npose)
                start_idx = clip_s_f_pose + i * self.args.stride
                fin_idx = start_idx + cut_length 
                sample_pose = pose_each_file[start_idx:fin_idx]

                sample_trans = trans_each_file[start_idx:fin_idx]
                sample_trans_v = trans_v_each_file[start_idx:fin_idx]
                sample_shape = shape_each_file[start_idx:fin_idx]
                # print(sample_pose.shape)
                if self.args.audio_rep is not None:
                    audio_start = clip_s_f_audio + math.floor(i * self.args.stride * self.args.audio_fps / self.args.pose_fps)
                    audio_end = audio_start + audio_short_length
                    sample_audio = audio_each_file[audio_start:audio_end]
                else:
                    sample_audio = np.array([-1])
                sample_facial = facial_each_file[start_idx:fin_idx] if self.args.facial_rep is not None else np.array([-1])
                sample_word = word_each_file[start_idx:fin_idx] if self.args.word_rep is not None else np.array([-1])
                sample_emo = emo_each_file[start_idx:fin_idx] if self.args.emo_rep is not None else np.array([-1])
                sample_sem = sem_each_file[start_idx:fin_idx] if self.args.sem_rep is not None else np.array([-1])
                sample_vid = vid_each_file[start_idx:fin_idx] if self.args.id_rep is not None else np.array([-1])
                
                if sample_pose.any() != None:
                    # filtering motion skeleton data
                    sample_pose, filtering_message = MotionPreprocessor(sample_pose).get()
                    is_correct_motion = (sample_pose is not None)
                    if is_correct_motion or disable_filtering:
                        sample_pose_list.append(sample_pose)
                        sample_audio_list.append(sample_audio)
                        sample_facial_list.append(sample_facial)
                        sample_shape_list.append(sample_shape)
                        sample_word_list.append(sample_word)
                        sample_vid_list.append(sample_vid)
                        sample_emo_list.append(sample_emo)
                        sample_sem_list.append(sample_sem)
                        sample_trans_list.append(sample_trans)
                        sample_trans_v_list.append(sample_trans_v)
                    else:
                        n_filtered_out[filtering_message] += 1

            if len(sample_pose_list) > 0:
                with dst_lmdb_env.begin(write=True) as txn:
                    for pose, audio, facial, shape, word, vid, emo, sem, trans,trans_v in zip(
                        sample_pose_list,
                        sample_audio_list,
                        sample_facial_list,
                        sample_shape_list,
                        sample_word_list,
                        sample_vid_list,
                        sample_emo_list,
                        sample_sem_list,
                        sample_trans_list,
                        sample_trans_v_list,):
                        k = "{:005}".format(self.n_out_samples).encode("ascii")
                        v = [pose, audio, facial, shape, word, emo, sem, vid, trans,trans_v]
                        v = pickle.dumps(v,5)
                        txn.put(k, v)
                        self.n_out_samples += 1
        return n_filtered_out

    def __getitem__(self, idx):
        with self.lmdb_env.begin(write=False) as txn:
            key = "{:005}".format(idx).encode("ascii")
            sample = txn.get(key)
            sample = pickle.loads(sample)
            tar_pose, in_audio, in_facial, in_shape, in_word, emo, sem, vid, trans,trans_v = sample
            #print(in_shape)
            #vid = torch.from_numpy(vid).int()
            emo = torch.from_numpy(emo).int()
            sem = torch.from_numpy(sem).float() 
            in_audio = torch.from_numpy(in_audio).float() 
            in_word = torch.from_numpy(in_word).float() if self.args.word_cache else torch.from_numpy(in_word).int() 
            if self.loader_type == "test":
                tar_pose = torch.from_numpy(tar_pose).float()
                trans = torch.from_numpy(trans).float()
                trans_v = torch.from_numpy(trans_v).float()
                in_facial = torch.from_numpy(in_facial).float()
                vid = torch.from_numpy(vid).float()
                in_shape = torch.from_numpy(in_shape).float()
            else:
                in_shape = torch.from_numpy(in_shape).reshape((in_shape.shape[0], -1)).float()
                trans = torch.from_numpy(trans).reshape((trans.shape[0], -1)).float()
                trans_v = torch.from_numpy(trans_v).reshape((trans_v.shape[0], -1)).float()
                vid = torch.from_numpy(vid).reshape((vid.shape[0], -1)).float()
                tar_pose = torch.from_numpy(tar_pose).reshape((tar_pose.shape[0], -1)).float()
                in_facial = torch.from_numpy(in_facial).reshape((in_facial.shape[0], -1)).float()
            return {"pose":tar_pose, "audio":in_audio, "facial":in_facial, "beta": in_shape, "word":in_word, "id":vid, "emo":emo, "sem":sem, "trans":trans,"trans_v":trans_v}

         
class MotionPreprocessor:
    def __init__(self, skeletons):
        self.skeletons = skeletons
        #self.mean_pose = mean_pose
        self.filtering_message = "PASS"

    def get(self):
        assert (self.skeletons is not None)

        # filtering
        if self.skeletons is not None:
            if self.check_pose_diff():
                self.skeletons = []
                self.filtering_message = "pose"
            # elif self.check_spine_angle():
            #     self.skeletons = []
            #     self.filtering_message = "spine angle"
            # elif self.check_static_motion():
            #     self.skeletons = []
            #     self.filtering_message = "motion"

        # if self.skeletons is not None:
        #     self.skeletons = self.skeletons.tolist()
        #     for i, frame in enumerate(self.skeletons):
        #         assert not np.isnan(self.skeletons[i]).any()  # missing joints

        return self.skeletons, self.filtering_message

    def check_static_motion(self, verbose=True):
        def get_variance(skeleton, joint_idx):
            wrist_pos = skeleton[:, joint_idx]
            variance = np.sum(np.var(wrist_pos, axis=0))
            return variance

        left_arm_var = get_variance(self.skeletons, 6)
        right_arm_var = get_variance(self.skeletons, 9)

        th = 0.0014  # exclude 13110
        # th = 0.002  # exclude 16905
        if left_arm_var < th and right_arm_var < th:
            if verbose:
                print("skip - check_static_motion left var {}, right var {}".format(left_arm_var, right_arm_var))
            return True
        else:
            if verbose:
                print("pass - check_static_motion left var {}, right var {}".format(left_arm_var, right_arm_var))
            return False


    def check_pose_diff(self, verbose=False):
#         diff = np.abs(self.skeletons - self.mean_pose) # 186*1
#         diff = np.mean(diff)

#         # th = 0.017
#         th = 0.02 #0.02  # exclude 3594
#         if diff < th:
#             if verbose:
#                 print("skip - check_pose_diff {:.5f}".format(diff))
#             return True
# #         th = 3.5 #0.02  # exclude 3594
# #         if 3.5 < diff < 5:
# #             if verbose:
# #                 print("skip - check_pose_diff {:.5f}".format(diff))
# #             return True
#         else:
#             if verbose:
#                 print("pass - check_pose_diff {:.5f}".format(diff))
        return False


    def check_spine_angle(self, verbose=True):
        def angle_between(v1, v2):
            v1_u = v1 / np.linalg.norm(v1)
            v2_u = v2 / np.linalg.norm(v2)
            return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

        angles = []
        for i in range(self.skeletons.shape[0]):
            spine_vec = self.skeletons[i, 1] - self.skeletons[i, 0]
            angle = angle_between(spine_vec, [0, -1, 0])
            angles.append(angle)

        if np.rad2deg(max(angles)) > 30 or np.rad2deg(np.mean(angles)) > 20:  # exclude 4495
        # if np.rad2deg(max(angles)) > 20:  # exclude 8270
            if verbose:
                print("skip - check_spine_angle {:.5f}, {:.5f}".format(max(angles), np.mean(angles)))
            return True
        else:
            if verbose:
                print("pass - check_spine_angle {:.5f}".format(max(angles)))
            return False
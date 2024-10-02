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

# ACCAD 120
# BioMotionLab_NTroje 120
# CMU 很复杂
# EKUT 100
# Eyes_Japan_Dataset 很复杂
# HumanEva 很复杂
# KIT 100
# MPI_HDM05 120
# MPI_Limits 120
# MPI_mosh 很复杂
# SFU 120
# SSM_synced 很复杂
# TCD_handMocap 很复杂
# TotalCapture 60
# Transitions_mocap 120

all_sequences = [
    'ACCAD',
    'BioMotionLab_NTroje',
    'CMU',
    'EKUT',
    'Eyes_Japan_Dataset',
    'HumanEva',
    'KIT',
    'MPI_HDM05',
    'MPI_Limits',
    'MPI_mosh',
    'SFU',
    'SSM_synced',
    'TCD_handMocap',
    'TotalCapture',
    'Transitions_mocap',
]
amass_test_split = ['Transitions_mocap', 'SSM_synced']
amass_vald_split = ['HumanEva', 'MPI_HDM05', 'SFU', 'MPI_mosh']
amass_train_split = ['BioMotionLab_NTroje', 'Eyes_Japan_Dataset', 'TotalCapture', 'KIT', 'ACCAD', 'CMU', 'MPI_Limits',
                     'TCD_handMocap', 'EKUT']

# 上面这些spilt方式是MOTION CLIP的，但是由于motionx中的framerate处理有问题，我先暂且只挑部分数据集进行训练
# 这些都是120fps的
# amass_test_split = ['SFU']
# amass_vald_split = ['MPI_Limits']
# amass_train_split = ['BioMotionLab_NTroje', 'MPI_HDM05', 'ACCAD','Transitions_mocap']


amass_splits = {
    'test': amass_test_split,
    'val': amass_vald_split,
    'train': amass_train_split
}
# assert len(amass_splits['train'] + amass_splits['test'] + amass_splits['vald']) == len(all_sequences) == 15

class CustomDataset(Dataset):
    def __init__(self, args, loader_type, augmentation=None, kwargs=None, build_cache=True):
        self.args = args
        self.loader_type = loader_type

        self.rank = dist.get_rank()
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
        self.use_amass = args.use_amass
        self.beatx_during_time = 0
        self.amass_during_time = 0
        
        if loader_type == "test": 
            self.args.multi_length_training = [1.0]
        self.max_length = int(args.pose_length * self.args.multi_length_training[-1])
        self.max_audio_pre_len = math.floor(args.pose_length / args.pose_fps * self.args.audio_sr)
        if self.max_audio_pre_len > self.args.test_length*self.args.audio_sr: 
            self.max_audio_pre_len = self.args.test_length*self.args.audio_sr
        
        if args.word_rep is not None:
            with open(f"{args.data_path}weights/vocab.pkl", 'rb') as f:
                self.lang_model = pickle.load(f)
                
        preloaded_dir = self.args.root_path + self.args.cache_path + loader_type + f"/{args.pose_rep}_cache"      
        # if args.pose_norm:
        #     # careful for rotation vectors
        #     if not os.path.exists(args.data_path+args.mean_pose_path+f"{args.pose_rep.split('_')[0]}/bvh_mean.npy"):
        #         self.calculate_mean_pose()
        #     self.mean_pose = np.load(args.data_path+args.mean_pose_path+f"{args.pose_rep.split('_')[0]}/bvh_mean.npy")
        #     self.std_pose = np.load(args.data_path+args.mean_pose_path+f"{args.pose_rep.split('_')[0]}/bvh_std.npy")
        # if args.audio_norm:
        #     if not os.path.exists(args.data_path+args.mean_pose_path+f"{args.audio_rep.split('_')[0]}/bvh_mean.npy"):
        #         self.calculate_mean_audio()
        #     self.mean_audio = np.load(args.data_path+args.mean_pose_path+f"{args.audio_rep.split('_')[0]}/npy_mean.npy")
        #     self.std_audio = np.load(args.data_path+args.mean_pose_path+f"{args.audio_rep.split('_')[0]}/npy_std.npy")
        # if args.facial_norm:
        #     if not os.path.exists(args.data_path+args.mean_pose_path+f"{args.pose_rep.split('_')[0]}/bvh_mean.npy"):
        #         self.calculate_mean_face()
        #     self.mean_facial = np.load(args.data_path+args.mean_pose_path+f"{args.facial_rep}/json_mean.npy")
        #     self.std_facial = np.load(args.data_path+args.mean_pose_path+f"{args.facial_rep}/json_std.npy")
        if self.args.beat_align:
            if not os.path.exists(args.data_path+f"weights/mean_vel_{args.pose_rep}.npy"):
                self.calculate_mean_velocity(args.data_path+f"weights/mean_vel_{args.pose_rep}.npy")
            self.avg_vel = np.load(args.data_path+f"weights/mean_vel_{args.pose_rep}.npy")
            
        if build_cache and self.rank == 0:
            self.build_cache(preloaded_dir)
        self.lmdb_env = lmdb.open(preloaded_dir, readonly=True, lock=False)
        with self.lmdb_env.begin() as txn:
            self.n_samples = txn.stat()["entries"] 

    def load_amass(self,data):
        ## 这个是用来
        # 修改amass数据里面的朝向，原本在blender里面是Z轴向上，目标是Y轴向上，当时面向目前没改
        
        data_dict = {key: data[key] for key in data}
        frames = data_dict['poses'].shape[0]
        b = data_dict['poses'][...,:3]
        b = rc.axis_angle_to_matrix(torch.from_numpy(b))
        rot_matrix = np.array([[1.0, 0.0, 0.0], [0.0 , 0.0, 1.0], [0.0, -1.0, 0.0]])
        c = np.einsum('ij,kjl->kil',rot_matrix,b)
        c = rc.matrix_to_axis_angle(torch.from_numpy(c))
        data_dict['poses'][...,:3] = c
        
        trans_matrix1 = np.array([[1.0, 0.0, 0.0], [0.0 , 0.0, -1.0], [0.0, 1.0, 0.0]])
        data_dict['trans'] = np.einsum("bi,ij->bj",data_dict['trans'],trans_matrix1)
        
        betas300 = np.zeros(300)
        betas300[:16] = data_dict['betas']
        data_dict['betas'] = betas300
        data_dict["expressions"] = np.zeros((frames,100))
        
        return data_dict


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
        logger.info(f"BEATX during time is {self.beatx_during_time}s !")
        logger.info(f"AMASS during time is {self.amass_during_time}s !")
        
        ## 对于BEATX train ,val ,test: 69800s ,7695s, 18672s ,总计 26.7h
        ##
        
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
        dst_lmdb_env = lmdb.open(out_lmdb_dir, map_size= int(1024 ** 3 * 50))# 50G
        n_filtered_out = defaultdict(int)
    
        for index, file_name in self.selected_file.iterrows():
            f_name = file_name["id"]
            ext = ".npz" if "smplx" in self.args.pose_rep else ".bvh"
            pose_file = self.data_dir + self.args.pose_rep + "/" + f_name + ext
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
            id_pose = f_name #1_wayne_0_1_1
            
            logger.info(colored(f"# ---- Building cache for Pose   {id_pose} ---- #", "blue"))
            if "smplx" in self.args.pose_rep:
                pose_data = np.load(pose_file, allow_pickle=True)
                assert 30%self.args.pose_fps == 0, 'pose_fps should be an aliquot part of 30'
                stride = int(30/self.args.pose_fps)
                pose_each_file = pose_data["poses"][::stride] * self.joint_mask
                pose_each_file = pose_each_file[:, self.joint_mask.astype(bool)]
                # print(pose_each_file.shape)
                self.beatx_during_time += pose_each_file.shape[0]/30
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
                if self.args.facial_rep is not None:
                    logger.info(f"# ---- Building cache for Facial {id_pose} and Pose {id_pose} ---- #")
                    facial_each_file = pose_data["expressions"][::stride]
                    if self.args.facial_norm: 
                        facial_each_file = (facial_each_file - self.mean_facial) / self.std_facial
                    
            if self.args.id_rep is not None:
                vid_each_file = np.repeat(np.array(int(f_name.split("_")[0])-1).reshape(1, 1), pose_each_file.shape[0], axis=0)
      
            filtered_result = self._sample_from_clip(
                dst_lmdb_env,
                pose_each_file, trans_each_file,trans_v_each_file, shape_each_file,
                vid_each_file,
                disable_filtering, clean_first_seconds, clean_final_seconds, is_test,
                ) 
            for type in filtered_result.keys():
                n_filtered_out[type] += filtered_result[type]
                    
        if self.args.use_amass:
            amass_dir = '/mnt/fu09a/chenbohong/PantoMatrix/scripts/EMAGE_2024/datasets/AMASS_SMPLX'
            for dataset in amass_splits[self.loader_type]:
                search_path = os.path.join(amass_dir,dataset, '**', '*.npz')
                npz_files = glob.glob(search_path, recursive=True)     
                for index, file_name in enumerate(npz_files):
                    f_name = file_name.split('/')[-1]
                    ext = ".npz" if "smplx" in self.args.pose_rep else ".bvh"
                    pose_file = file_name
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
                    id_pose = f_name #1_wayne_0_1_1
                    
                    logger.info(colored(f"# ---- Building cache for Pose   {id_pose} ---- #", "blue"))
                    if "smplx" in self.args.pose_rep:
                        pose_data = np.load(pose_file, allow_pickle=True)
                        if len(pose_data.files)==6:
                            logger.info(colored(f"# ---- state file ---- #", "red"))
                            continue
                        assert 30%self.args.pose_fps == 0, 'pose_fps should be an aliquot part of 30'
                        pose_each_file = self.load_amass(pose_data)
                        fps = pose_data['mocap_frame_rate']
                        stride =round(fps/30)
                        pose_each_file = pose_data["poses"][::stride] * self.joint_mask
                        pose_each_file = pose_each_file[:, self.joint_mask.astype(bool)]
                        trans_each_file = pose_data["trans"][::stride]
                        

                        trans_each_file[:,0] = trans_each_file[:,0] - trans_each_file[0,0]
                        trans_each_file[:,2] = trans_each_file[:,2] - trans_each_file[0,2]
                        trans_v_each_file = np.zeros_like(trans_each_file)
                        trans_v_each_file[1:,0] = trans_each_file[1:,0] - trans_each_file[:-1,0]
                        trans_v_each_file[0,0] = trans_v_each_file[1,0]
                        trans_v_each_file[1:,2] = trans_each_file[1:,2] - trans_each_file[:-1,2]
                        trans_v_each_file[0,2] = trans_v_each_file[1,2]
                        trans_v_each_file[:,1] = trans_each_file[:,1]
                        
                        
                        
                        shape_each_file = np.repeat(pose_data["betas"].reshape(1, -1), pose_each_file.shape[0], axis=0)
                                
                    if self.args.id_rep is not None:
                        vid_each_file = np.repeat(np.array(int(100)-1).reshape(1, 1), pose_each_file.shape[0], axis=0)
            
                    filtered_result = self._sample_from_clip(
                        dst_lmdb_env,
                        pose_each_file, trans_each_file,trans_v_each_file, shape_each_file,
                        vid_each_file,
                        disable_filtering, clean_first_seconds, clean_final_seconds, is_test,
                        ) 
                    for type in filtered_result.keys():
                        n_filtered_out[type] += filtered_result[type]
                    
                    
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
        self, dst_lmdb_env, pose_each_file, trans_each_file, trans_v_each_file,shape_each_file,
        vid_each_file,
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
            

            n_filtered_out = defaultdict(int)
            sample_pose_list = []
            sample_audio_list = []
            sample_shape_list = []
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


                sample_vid = vid_each_file[start_idx:fin_idx] if self.args.id_rep is not None else np.array([-1])
                
                if sample_pose.any() != None:
                    # filtering motion skeleton data
                    sample_pose, filtering_message = MotionPreprocessor(sample_pose).get()
                    is_correct_motion = (sample_pose != [])
                    if is_correct_motion or disable_filtering:
                        sample_pose_list.append(sample_pose)

                        sample_shape_list.append(sample_shape)

                        sample_vid_list.append(sample_vid)


                        sample_trans_list.append(sample_trans)
                        sample_trans_v_list.append(sample_trans_v)
                    else:
                        n_filtered_out[filtering_message] += 1

            if len(sample_pose_list) > 0:
                with dst_lmdb_env.begin(write=True) as txn:
                    for pose, shape,  vid, trans,trans_v in zip(
                        sample_pose_list,
                        sample_shape_list,
                        sample_vid_list,
                        sample_trans_list,
                        sample_trans_v_list,
                        ):
                        k = "{:005}".format(self.n_out_samples).encode("ascii")
                        v = [pose , shape, vid, trans,trans_v]
                        v = pickle.dumps(v,5)
                        txn.put(k, v)
                        self.n_out_samples += 1
        return n_filtered_out

    def __getitem__(self, idx):
        with self.lmdb_env.begin(write=False) as txn:
            key = "{:005}".format(idx).encode("ascii")
            sample = txn.get(key)
            sample = pickle.loads(sample)
            tar_pose,  in_shape, vid, trans, trans_v = sample

            if self.loader_type == "test":
                tar_pose = torch.from_numpy(tar_pose).float()
                trans = torch.from_numpy(trans).float()
                trans_v = torch.from_numpy(trans_v).float()
                vid = torch.from_numpy(vid).float()
                in_shape = torch.from_numpy(in_shape).float()
            else:
                in_shape = torch.from_numpy(in_shape).reshape((in_shape.shape[0], -1)).float()
                trans = torch.from_numpy(trans).reshape((trans.shape[0], -1)).float()
                trans_v = torch.from_numpy(trans_v).reshape((trans_v.shape[0], -1)).float()
                vid = torch.from_numpy(vid).reshape((vid.shape[0], -1)).float()
                tar_pose = torch.from_numpy(tar_pose).reshape((tar_pose.shape[0], -1)).float()
            return {"pose":tar_pose, "trans":trans,"trans_v":trans_v, "vid":vid}

         
class MotionPreprocessor:
    def __init__(self, skeletons):
        self.skeletons = skeletons
        #self.mean_pose = mean_pose
        self.filtering_message = "PASS"

    def get(self):
        assert (self.skeletons is not None)

        # filtering
        if self.skeletons != []:
            if self.check_pose_diff():
                self.skeletons = []
                self.filtering_message = "pose"
            # elif self.check_spine_angle():
            #     self.skeletons = []
            #     self.filtering_message = "spine angle"
            # elif self.check_static_motion():
            #     self.skeletons = []
            #     self.filtering_message = "motion"

        # if self.skeletons != []:
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
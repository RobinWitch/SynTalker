import os
import signal
import time
import csv
import sys
import warnings
import random
import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import numpy as np
import time
import pprint
from loguru import logger
import smplx
from torch.utils.tensorboard import SummaryWriter
import wandb
import matplotlib.pyplot as plt
from utils import config, logger_tools, other_tools_hf, metric, data_transfer, other_tools
from dataloaders import data_tools
from dataloaders.build_vocab import Vocab
from optimizers.optim_factory import create_optimizer
from optimizers.scheduler_factory import create_scheduler
from optimizers.loss_factory import get_loss_func
from dataloaders.data_tools import joints_list
from utils import rotation_conversions as rc
import soundfile as sf
import librosa 
import subprocess
from transformers import pipeline
from diffusion.model_util import create_gaussian_diffusion
from diffusion.resample import create_named_schedule_sampler
from models.vq.model import RVQVAE
import train

device = "cuda:0" if torch.cuda.is_available() else "cpu"

pipe = pipeline(
  "automatic-speech-recognition",
  model="openai/whisper-tiny.en",
  chunk_length_s=30,
  device=device,
)

debug = False

class BaseTrainer(object):
    def __init__(self, args,ap):
        args.use_ddim=True
        hf_dir = "hf"
        tmp_dir = args.out_path + "custom/" + hf_dir
        if not os.path.exists(tmp_dir + "/"):
            os.makedirs(tmp_dir + "/")
        self.audio_path = tmp_dir + "/tmp.wav"
        sf.write(self.audio_path, ap[1], ap[0])
        
        
        audio, ssr = librosa.load(self.audio_path,sr=args.audio_sr)

        # use asr model to get corresponding text transcripts
        file_path = tmp_dir+"/tmp.lab"
        self.textgrid_path = tmp_dir + "/tmp.TextGrid"
        if not debug:
            text = pipe(audio, batch_size=8)["text"]
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(text)
            
            # use montreal forced aligner to get textgrid
            
            command = ["mfa", "align", tmp_dir, "english_us_arpa", "english_us_arpa", tmp_dir]
            result = subprocess.run(command, capture_output=True, text=True)
            

        ap = (ssr, audio)
        self.args = args
        self.rank = 0 # dist.get_rank()
       
        args.textgrid_file_path = self.textgrid_path
        args.audio_file_path = self.audio_path
    
    
        self.rank = 0 # dist.get_rank()
       
        self.checkpoint_path = args.out_path + "custom/" + hf_dir + "/" 
        if self.rank == 0:
            self.test_data = __import__(f"dataloaders.{args.dataset}", fromlist=["something"]).CustomDataset(args, "test")
            self.test_loader = torch.utils.data.DataLoader(
                self.test_data, 
                batch_size=1,  
                shuffle=False,  
                num_workers=args.loader_workers,
                drop_last=False,
            )
        logger.info(f"Init test dataloader success")
        model_module = __import__(f"models.{args.model}", fromlist=["something"])
        
        self.model = torch.nn.DataParallel(getattr(model_module, args.g_name)(args), args.gpus).cuda()
        
        if self.rank == 0:
            logger.info(self.model)
            logger.info(f"init {args.g_name} success")

        self.smplx = smplx.create(
        self.args.data_path_1+"smplx_models/", 
            model_type='smplx',
            gender='NEUTRAL_2020', 
            use_face_contour=False,
            num_betas=300,
            num_expression_coeffs=100, 
            ext='npz',
            use_pca=False,
        ).to(self.rank).eval()    



    
    
        self.args = args
        self.ori_joint_list = joints_list[self.args.ori_joints]
        self.tar_joint_list_face = joints_list["beat_smplx_face"]
        self.tar_joint_list_upper = joints_list["beat_smplx_upper"]
        self.tar_joint_list_hands = joints_list["beat_smplx_hands"]
        self.tar_joint_list_lower = joints_list["beat_smplx_lower"]
       
        self.joint_mask_face = np.zeros(len(list(self.ori_joint_list.keys()))*3)
        self.joints = 55
        for joint_name in self.tar_joint_list_face:
            self.joint_mask_face[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1
        self.joint_mask_upper = np.zeros(len(list(self.ori_joint_list.keys()))*3)
        for joint_name in self.tar_joint_list_upper:
            self.joint_mask_upper[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1
        self.joint_mask_hands = np.zeros(len(list(self.ori_joint_list.keys()))*3)
        for joint_name in self.tar_joint_list_hands:
            self.joint_mask_hands[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1
        self.joint_mask_lower = np.zeros(len(list(self.ori_joint_list.keys()))*3)
        for joint_name in self.tar_joint_list_lower:
            self.joint_mask_lower[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1

        self.tracker = other_tools.EpochTracker(["fid", "l1div", "bc", "rec", "trans", "vel", "transv", 'dis', 'gen', 'acc', 'transa', 'exp', 'lvd', 'mse', "cls", "rec_face", "latent", "cls_full", "cls_self", "cls_word", "latent_word","latent_self","predict_x0_loss"], [False,True,True, False, False, False, False, False, False, False, False, False, False, False, False, False, False,False, False, False,False,False,False])
            
        vq_model_module = __import__(f"models.motion_representation", fromlist=["something"])
        self.args.vae_layer = 2
        self.args.vae_length = 256
        self.args.vae_test_dim = 106
        self.vq_model_face = getattr(vq_model_module, "VQVAEConvZero")(self.args).to(self.rank)
        other_tools.load_checkpoints(self.vq_model_face, "./datasets/hub/pretrained_vq/face_vertex_1layer_790.bin", args.e_name)
        
        
        vq_type = self.args.vqvae_type
        if vq_type=="vqvae":
            
            self.args.vae_layer = 4
            self.args.vae_test_dim = 78
            self.vq_model_upper = getattr(vq_model_module, "VQVAEConvZero")(self.args).to(self.rank)
            other_tools.load_checkpoints(self.vq_model_upper, args.vqvae_upper_path, args.e_name)
            self.args.vae_test_dim = 180
            self.vq_model_hands = getattr(vq_model_module, "VQVAEConvZero")(self.args).to(self.rank)
            other_tools.load_checkpoints(self.vq_model_hands, args.vqvae_hands_path, args.e_name)
            self.args.vae_test_dim = 54
            self.args.vae_layer = 4
            self.vq_model_lower = getattr(vq_model_module, "VQVAEConvZero")(self.args).to(self.rank)
            other_tools.load_checkpoints(self.vq_model_lower, args.vqvae_lower_path, args.e_name)
            
        elif vq_type=="rvqvae":

            args.num_quantizers = 6
            args.shared_codebook =  False
            args.quantize_dropout_prob = 0.2
            args.mu = 0.99

            args.nb_code = 512
            args.code_dim = 512
            args.code_dim = 512
            args.down_t = 2
            args.stride_t = 2
            args.width = 512
            args.depth = 3
            args.dilation_growth_rate = 3
            args.vq_act = "relu"
            args.vq_norm = None
                                
            dim_pose = 78  
            args.body_part = "upper"
            self.vq_model_upper = RVQVAE(args,
                                dim_pose,
                                args.nb_code,
                                args.code_dim,
                                args.code_dim,
                                args.down_t,
                                args.stride_t,
                                args.width,
                                args.depth,
                                args.dilation_growth_rate,
                                args.vq_act,
                                args.vq_norm)

            dim_pose = 180
            args.body_part = "hands"
            self.vq_model_hands = RVQVAE(args,
                                dim_pose,
                                args.nb_code,
                                args.code_dim,
                                args.code_dim,
                                args.down_t,
                                args.stride_t,
                                args.width,
                                args.depth,
                                args.dilation_growth_rate,
                                args.vq_act,
                                args.vq_norm)

            dim_pose = 54
            if args.use_trans:
                dim_pose = 57
                self.args.vqvae_lower_path = self.args.vqvae_lower_trans_path
            args.body_part = "lower"
            self.vq_model_lower = RVQVAE(args,
                                dim_pose,
                                args.nb_code,
                                args.code_dim,
                                args.code_dim,
                                args.down_t,
                                args.stride_t,
                                args.width,
                                args.depth,
                                args.dilation_growth_rate,
                                args.vq_act,
                                args.vq_norm)
                    
            self.vq_model_upper.load_state_dict(torch.load(self.args.vqvae_upper_path)['net'])
            self.vq_model_hands.load_state_dict(torch.load(self.args.vqvae_hands_path)['net'])
            self.vq_model_lower.load_state_dict(torch.load(self.args.vqvae_lower_path)['net'])
            
            self.vqvae_latent_scale = self.args.vqvae_latent_scale 

            self.vq_model_upper.eval().to(self.rank)
            self.vq_model_hands.eval().to(self.rank)
            self.vq_model_lower.eval().to(self.rank)
            
        
        
        
        
        self.args.vae_test_dim = 61
        self.args.vae_layer = 4
        self.args.vae_test_dim = 330
        self.args.vae_layer = 4
        self.args.vae_length = 240


        self.vq_model_face.eval()
        self.vq_model_upper.eval()
        self.vq_model_hands.eval()
        self.vq_model_lower.eval()

        self.cls_loss = nn.NLLLoss().to(self.rank)
        self.reclatent_loss = nn.MSELoss().to(self.rank)
        self.vel_loss = torch.nn.L1Loss(reduction='mean').to(self.rank)
        self.rec_loss = get_loss_func("GeodesicLoss").to(self.rank) 
        self.log_softmax = nn.LogSoftmax(dim=2).to(self.rank)
        
        self.diffusion = create_gaussian_diffusion(use_ddim=args.use_ddim)
        self.schedule_sampler_type = 'uniform'
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, self.diffusion)
        self.mean = np.load(args.mean_pose_path)
        self.std = np.load(args.std_pose_path)
        
        self.use_trans = args.use_trans
        if self.use_trans:
            self.trans_mean = np.load(args.mean_trans_path)
            self.trans_std = np.load(args.std_trans_path)
            self.trans_mean = torch.from_numpy(self.trans_mean).cuda()
            self.trans_std = torch.from_numpy(self.trans_std).cuda()
        

        joints = [3,6,9,12,13,14,15,16,17,18,19,20,21]
        upper_body_mask = []
        for i in joints:
            upper_body_mask.extend([i*6, i*6+1, i*6+2, i*6+3, i*6+4, i*6+5])

        joints = list(range(25,55))
        hands_body_mask = []
        for i in joints:
            hands_body_mask.extend([i*6, i*6+1, i*6+2, i*6+3, i*6+4, i*6+5])

        joints = [0,1,2,4,5,7,8,10,11]
        lower_body_mask = []
        for i in joints:
            lower_body_mask.extend([i*6, i*6+1, i*6+2, i*6+3, i*6+4, i*6+5])

        self.mean_upper = self.mean[upper_body_mask]
        self.mean_hands = self.mean[hands_body_mask]
        self.mean_lower = self.mean[lower_body_mask]
        self.std_upper = self.std[upper_body_mask]
        self.std_hands = self.std[hands_body_mask]
        self.std_lower = self.std[lower_body_mask]
        
        self.mean_upper = torch.from_numpy(self.mean_upper).cuda()
        self.mean_hands = torch.from_numpy(self.mean_hands).cuda()
        self.mean_lower = torch.from_numpy(self.mean_lower).cuda()
        self.std_upper = torch.from_numpy(self.std_upper).cuda()
        self.std_hands = torch.from_numpy(self.std_hands).cuda()
        self.std_lower = torch.from_numpy(self.std_lower).cuda()
      
    
    def inverse_selection(self, filtered_t, selection_array, n):
        original_shape_t = np.zeros((n, selection_array.size))
        selected_indices = np.where(selection_array == 1)[0]
        for i in range(n):
            original_shape_t[i, selected_indices] = filtered_t[i]
        return original_shape_t
    
    def inverse_selection_tensor(self, filtered_t, selection_array, n):
        selection_array = torch.from_numpy(selection_array).cuda()
        original_shape_t = torch.zeros((n, 165)).cuda()
        selected_indices = torch.where(selection_array == 1)[0]
        for i in range(n):
            original_shape_t[i, selected_indices] = filtered_t[i]
        return original_shape_t
    
    def _load_data(self, dict_data):
        tar_pose_raw = dict_data["pose"]
        tar_pose = tar_pose_raw[:, :, :165].to(self.rank)
        tar_contact = tar_pose_raw[:, :, 165:169].to(self.rank)
        tar_trans = dict_data["trans"].to(self.rank)
        tar_trans_v = dict_data["trans_v"].to(self.rank)
        tar_exps = dict_data["facial"].to(self.rank)
        in_audio = dict_data["audio"].to(self.rank) 
        in_word = dict_data["word"].to(self.rank)
        tar_beta = dict_data["beta"].to(self.rank)
        tar_id = dict_data["id"].to(self.rank).long()
        bs, n, j = tar_pose.shape[0], tar_pose.shape[1], self.joints

        tar_pose_jaw = tar_pose[:, :, 66:69]
        tar_pose_jaw = rc.axis_angle_to_matrix(tar_pose_jaw.reshape(bs, n, 1, 3))
        tar_pose_jaw = rc.matrix_to_rotation_6d(tar_pose_jaw).reshape(bs, n, 1*6)
        tar_pose_face = torch.cat([tar_pose_jaw, tar_exps], dim=2)

        tar_pose_hands = tar_pose[:, :, 25*3:55*3]
        tar_pose_hands = rc.axis_angle_to_matrix(tar_pose_hands.reshape(bs, n, 30, 3))
        tar_pose_hands = rc.matrix_to_rotation_6d(tar_pose_hands).reshape(bs, n, 30*6)

        tar_pose_upper = tar_pose[:, :, self.joint_mask_upper.astype(bool)]
        tar_pose_upper = rc.axis_angle_to_matrix(tar_pose_upper.reshape(bs, n, 13, 3))
        tar_pose_upper = rc.matrix_to_rotation_6d(tar_pose_upper).reshape(bs, n, 13*6)

        tar_pose_leg = tar_pose[:, :, self.joint_mask_lower.astype(bool)]
        tar_pose_leg = rc.axis_angle_to_matrix(tar_pose_leg.reshape(bs, n, 9, 3))
        tar_pose_leg = rc.matrix_to_rotation_6d(tar_pose_leg).reshape(bs, n, 9*6)

        tar_pose_lower = tar_pose_leg


        tar4dis = torch.cat([tar_pose_jaw, tar_pose_upper, tar_pose_hands, tar_pose_leg], dim=2)

      
        if self.args.pose_norm:
            tar_pose_upper = (tar_pose_upper - self.mean_upper) / self.std_upper
            tar_pose_hands = (tar_pose_hands - self.mean_hands) / self.std_hands
            tar_pose_lower = (tar_pose_lower - self.mean_lower) / self.std_lower
        
        if self.use_trans:
            tar_trans_v = (tar_trans_v - self.trans_mean)/self.trans_std
            tar_pose_lower = torch.cat([tar_pose_lower,tar_trans_v], dim=-1)
      
        latent_face_top = self.vq_model_face.map2latent(tar_pose_face) # bs*n/4
        latent_upper_top = self.vq_model_upper.map2latent(tar_pose_upper)
        latent_hands_top = self.vq_model_hands.map2latent(tar_pose_hands)
        latent_lower_top = self.vq_model_lower.map2latent(tar_pose_lower)
        
        latent_in = torch.cat([latent_upper_top, latent_hands_top, latent_lower_top], dim=2)/self.args.vqvae_latent_scale
        
        
        tar_pose_6d = rc.axis_angle_to_matrix(tar_pose.reshape(bs, n, 55, 3))
        tar_pose_6d = rc.matrix_to_rotation_6d(tar_pose_6d).reshape(bs, n, 55*6)
        latent_all = torch.cat([tar_pose_6d, tar_trans, tar_contact], dim=-1)
        style_feature = None
        if self.args.use_motionclip:
            motionclip_feat = tar_pose_6d[...,:22*6]
            batch = {}
            bs,seq,feat = motionclip_feat.shape
            batch['x']=motionclip_feat.permute(0,2,1).contiguous()
            batch['y']=torch.zeros(bs).int().cuda()
            batch['mask']=torch.ones([bs,seq]).bool().cuda()
            style_feature = self.motionclip.encoder(batch)['mu'].detach().float()
            
        
        
        # print(tar_index_value_upper_top.shape, index_in.shape)
        return {
            "tar_pose_jaw": tar_pose_jaw,
            "tar_pose_face": tar_pose_face,
            "tar_pose_upper": tar_pose_upper,
            "tar_pose_lower": tar_pose_lower,
            "tar_pose_hands": tar_pose_hands,
            'tar_pose_leg': tar_pose_leg,
            "in_audio": in_audio,
            "in_word": in_word,
            "tar_trans": tar_trans,
            "tar_exps": tar_exps,
            "tar_beta": tar_beta,
            "tar_pose": tar_pose,
            "tar4dis": tar4dis,
            "latent_face_top": latent_face_top,
            "latent_upper_top": latent_upper_top,
            "latent_hands_top": latent_hands_top,
            "latent_lower_top": latent_lower_top,
            "latent_in":  latent_in,
            "tar_id": tar_id,
            "latent_all": latent_all,
            "tar_pose_6d": tar_pose_6d,
            "tar_contact": tar_contact,
            "style_feature":style_feature,
        }
    
    def _g_test(self, loaded_data):
        sample_fn = self.diffusion.p_sample_loop
        if self.args.use_ddim:
            sample_fn = self.diffusion.ddim_sample_loop
        mode = 'test'
        bs, n, j = loaded_data["tar_pose"].shape[0], loaded_data["tar_pose"].shape[1], self.joints 
        tar_pose = loaded_data["tar_pose"]
        tar_beta = loaded_data["tar_beta"]
        tar_exps = loaded_data["tar_exps"]
        tar_contact = loaded_data["tar_contact"]
        tar_trans = loaded_data["tar_trans"]
        in_word = loaded_data["in_word"]
        in_audio = loaded_data["in_audio"]
        in_x0 = loaded_data['latent_in']
        in_seed = loaded_data['latent_in']
        
        remain = n%8
        if remain != 0:
            tar_pose = tar_pose[:, :-remain, :]
            tar_beta = tar_beta[:, :-remain, :]
            tar_trans = tar_trans[:, :-remain, :]
            in_word = in_word[:, :-remain]
            tar_exps = tar_exps[:, :-remain, :]
            tar_contact = tar_contact[:, :-remain, :]
            in_x0 = in_x0[:, :in_x0.shape[1]-(remain//self.args.vqvae_squeeze_scale), :]
            in_seed = in_seed[:, :in_x0.shape[1]-(remain//self.args.vqvae_squeeze_scale), :]
            n = n - remain

        tar_pose_jaw = tar_pose[:, :, 66:69]
        tar_pose_jaw = rc.axis_angle_to_matrix(tar_pose_jaw.reshape(bs, n, 1, 3))
        tar_pose_jaw = rc.matrix_to_rotation_6d(tar_pose_jaw).reshape(bs, n, 1*6)
        tar_pose_face = torch.cat([tar_pose_jaw, tar_exps], dim=2)

        tar_pose_hands = tar_pose[:, :, 25*3:55*3]
        tar_pose_hands = rc.axis_angle_to_matrix(tar_pose_hands.reshape(bs, n, 30, 3))
        tar_pose_hands = rc.matrix_to_rotation_6d(tar_pose_hands).reshape(bs, n, 30*6)

        tar_pose_upper = tar_pose[:, :, self.joint_mask_upper.astype(bool)]
        tar_pose_upper = rc.axis_angle_to_matrix(tar_pose_upper.reshape(bs, n, 13, 3))
        tar_pose_upper = rc.matrix_to_rotation_6d(tar_pose_upper).reshape(bs, n, 13*6)

        tar_pose_leg = tar_pose[:, :, self.joint_mask_lower.astype(bool)]
        tar_pose_leg = rc.axis_angle_to_matrix(tar_pose_leg.reshape(bs, n, 9, 3))
        tar_pose_leg = rc.matrix_to_rotation_6d(tar_pose_leg).reshape(bs, n, 9*6)
        tar_pose_lower = torch.cat([tar_pose_leg, tar_trans, tar_contact], dim=2)
        
        tar_pose_6d = rc.axis_angle_to_matrix(tar_pose.reshape(bs, n, 55, 3))
        tar_pose_6d = rc.matrix_to_rotation_6d(tar_pose_6d).reshape(bs, n, 55*6)
        latent_all = torch.cat([tar_pose_6d, tar_trans, tar_contact], dim=-1)
        
        rec_all_face = []
        rec_all_upper = []
        rec_all_lower = []
        rec_all_hands = []
        vqvae_squeeze_scale = self.args.vqvae_squeeze_scale
        roundt = (n - self.args.pre_frames * vqvae_squeeze_scale) // (self.args.pose_length - self.args.pre_frames * vqvae_squeeze_scale)
        remain = (n - self.args.pre_frames * vqvae_squeeze_scale) % (self.args.pose_length - self.args.pre_frames * vqvae_squeeze_scale)
        round_l = self.args.pose_length - self.args.pre_frames * vqvae_squeeze_scale
         

        for i in range(0, roundt):
            in_word_tmp = in_word[:, i*(round_l):(i+1)*(round_l)+self.args.pre_frames * vqvae_squeeze_scale]

            in_audio_tmp = in_audio[:, i*(16000//30*round_l):(i+1)*(16000//30*round_l)+16000//30*self.args.pre_frames * vqvae_squeeze_scale]
            in_id_tmp = loaded_data['tar_id'][:, i*(round_l):(i+1)*(round_l)+self.args.pre_frames]
            in_seed_tmp = in_seed[:, i*(round_l)//vqvae_squeeze_scale:(i+1)*(round_l)//vqvae_squeeze_scale+self.args.pre_frames]
            in_x0_tmp = in_x0[:, i*(round_l)//vqvae_squeeze_scale:(i+1)*(round_l)//vqvae_squeeze_scale+self.args.pre_frames]
            mask_val = torch.ones(bs, self.args.pose_length, self.args.pose_dims+3+4).float().cuda()
            mask_val[:, :self.args.pre_frames, :] = 0.0
            if i == 0:
                in_seed_tmp = in_seed_tmp[:, :self.args.pre_frames, :]
            else:
                in_seed_tmp = last_sample[:, -self.args.pre_frames:, :]

            cond_ = {'y':{}}
            cond_['y']['audio'] = in_audio_tmp
            cond_['y']['word'] = in_word_tmp
            cond_['y']['id'] = in_id_tmp
            cond_['y']['seed'] =in_seed_tmp
            cond_['y']['mask'] = (torch.zeros([self.args.batch_size, 1, 1, self.args.pose_length]) < 1).cuda()
            

        
            cond_['y']['style_feature'] = torch.zeros([bs, 512]).cuda()

            shape_ = (bs, 1536, 1, 32)
            sample = sample_fn(
                self.model,
                shape_,
                clip_denoised=False,
                model_kwargs=cond_,
                skip_timesteps=0,
                init_image=None,
                progress=True,
                dump_steps=None,
                noise=None,
                const_noise=False,
            )
            sample = sample.squeeze().permute(1,0).unsqueeze(0)

            last_sample = sample.clone()
            
            rec_latent_upper = sample[...,:512]
            rec_latent_hands = sample[...,512:1024]
            rec_latent_lower = sample[...,1024:1536]
            
           

            if i == 0:
                rec_all_upper.append(rec_latent_upper)
                rec_all_hands.append(rec_latent_hands)
                rec_all_lower.append(rec_latent_lower)
            else:
                rec_all_upper.append(rec_latent_upper[:, self.args.pre_frames:])
                rec_all_hands.append(rec_latent_hands[:, self.args.pre_frames:])
                rec_all_lower.append(rec_latent_lower[:, self.args.pre_frames:])

        rec_all_upper = torch.cat(rec_all_upper, dim=1) * self.vqvae_latent_scale
        rec_all_hands = torch.cat(rec_all_hands, dim=1) * self.vqvae_latent_scale
        rec_all_lower = torch.cat(rec_all_lower, dim=1) * self.vqvae_latent_scale

        rec_upper = self.vq_model_upper.latent2origin(rec_all_upper)[0]
        rec_hands = self.vq_model_hands.latent2origin(rec_all_hands)[0]
        rec_lower = self.vq_model_lower.latent2origin(rec_all_lower)[0]
        
        
        if self.use_trans:
            rec_trans_v = rec_lower[...,-3:]
            rec_trans_v = rec_trans_v * self.trans_std + self.trans_mean
            rec_trans = torch.zeros_like(rec_trans_v)
            rec_trans = torch.cumsum(rec_trans_v, dim=-2)
            rec_trans[...,1]=rec_trans_v[...,1]
            rec_lower = rec_lower[...,:-3]
        
        if self.args.pose_norm:
            rec_upper = rec_upper * self.std_upper + self.mean_upper
            rec_hands = rec_hands * self.std_hands + self.mean_hands
            rec_lower = rec_lower * self.std_lower + self.mean_lower




        n = n - remain
        tar_pose = tar_pose[:, :n, :]
        tar_exps = tar_exps[:, :n, :]
        tar_trans = tar_trans[:, :n, :]
        tar_beta = tar_beta[:, :n, :]


        rec_exps = tar_exps
        #rec_pose_jaw = rec_face[:, :, :6]
        rec_pose_legs = rec_lower[:, :, :54]
        bs, n = rec_pose_legs.shape[0], rec_pose_legs.shape[1]
        rec_pose_upper = rec_upper.reshape(bs, n, 13, 6)
        rec_pose_upper = rc.rotation_6d_to_matrix(rec_pose_upper)#
        rec_pose_upper = rc.matrix_to_axis_angle(rec_pose_upper).reshape(bs*n, 13*3)
        rec_pose_upper_recover = self.inverse_selection_tensor(rec_pose_upper, self.joint_mask_upper, bs*n)
        rec_pose_lower = rec_pose_legs.reshape(bs, n, 9, 6)
        rec_pose_lower = rc.rotation_6d_to_matrix(rec_pose_lower)
        rec_lower2global = rc.matrix_to_rotation_6d(rec_pose_lower.clone()).reshape(bs, n, 9*6)
        rec_pose_lower = rc.matrix_to_axis_angle(rec_pose_lower).reshape(bs*n, 9*3)
        rec_pose_lower_recover = self.inverse_selection_tensor(rec_pose_lower, self.joint_mask_lower, bs*n)
        rec_pose_hands = rec_hands.reshape(bs, n, 30, 6)
        rec_pose_hands = rc.rotation_6d_to_matrix(rec_pose_hands)
        rec_pose_hands = rc.matrix_to_axis_angle(rec_pose_hands).reshape(bs*n, 30*3)
        rec_pose_hands_recover = self.inverse_selection_tensor(rec_pose_hands, self.joint_mask_hands, bs*n)
        rec_pose = rec_pose_upper_recover + rec_pose_lower_recover + rec_pose_hands_recover 
        rec_pose[:, 66:69] = tar_pose.reshape(bs*n, 55*3)[:, 66:69]

        rec_pose = rc.axis_angle_to_matrix(rec_pose.reshape(bs*n, j, 3))
        rec_pose = rc.matrix_to_rotation_6d(rec_pose).reshape(bs, n, j*6)
        tar_pose = rc.axis_angle_to_matrix(tar_pose.reshape(bs*n, j, 3))
        tar_pose = rc.matrix_to_rotation_6d(tar_pose).reshape(bs, n, j*6)
        
        return {
            'rec_pose': rec_pose,
            'rec_trans': rec_trans,
            'tar_pose': tar_pose,
            'tar_exps': tar_exps,
            'tar_beta': tar_beta,
            'tar_trans': tar_trans,
            'rec_exps': rec_exps,
        }


    def test_demo(self, epoch):
        '''
        input audio and text, output motion
        do not calculate loss and metric
        save video
        '''
        results_save_path = self.checkpoint_path + f"/{epoch}/"
        if os.path.exists(results_save_path): 
            import shutil
            shutil.rmtree(results_save_path)
        os.makedirs(results_save_path)
        start_time = time.time()
        total_length = 0
        test_seq_list = self.test_data.selected_file
        align = 0 
        latent_out = []
        latent_ori = []
        l2_all = 0 
        lvel = 0
        self.model.eval()
        self.smplx.eval()
        # self.eval_copy.eval()
        with torch.no_grad():
            for its, batch_data in enumerate(self.test_loader):
                loaded_data = self._load_data(batch_data)    
                net_out = self._g_test(loaded_data)
                tar_pose = net_out['tar_pose']
                rec_pose = net_out['rec_pose']
                tar_exps = net_out['tar_exps']
                tar_beta = net_out['tar_beta']
                rec_trans = net_out['rec_trans']
                tar_trans = net_out['tar_trans']
                rec_exps = net_out['rec_exps']
                bs, n, j = tar_pose.shape[0], tar_pose.shape[1], self.joints
                if (30/self.args.pose_fps) != 1:
                    assert 30%self.args.pose_fps == 0
                    n *= int(30/self.args.pose_fps)
                    tar_pose = torch.nn.functional.interpolate(tar_pose.permute(0, 2, 1), scale_factor=30/self.args.pose_fps, mode='linear').permute(0,2,1)
                    rec_pose = torch.nn.functional.interpolate(rec_pose.permute(0, 2, 1), scale_factor=30/self.args.pose_fps, mode='linear').permute(0,2,1)
                

                rec_pose = rc.rotation_6d_to_matrix(rec_pose.reshape(bs*n, j, 6))
                rec_pose = rc.matrix_to_rotation_6d(rec_pose).reshape(bs, n, j*6)
                tar_pose = rc.rotation_6d_to_matrix(tar_pose.reshape(bs*n, j, 6))
                tar_pose = rc.matrix_to_rotation_6d(tar_pose).reshape(bs, n, j*6)

                rec_pose = rc.rotation_6d_to_matrix(rec_pose.reshape(bs*n, j, 6))
                rec_pose = rc.matrix_to_axis_angle(rec_pose).reshape(bs*n, j*3)
                tar_pose = rc.rotation_6d_to_matrix(tar_pose.reshape(bs*n, j, 6))
                tar_pose = rc.matrix_to_axis_angle(tar_pose).reshape(bs*n, j*3)
                

                tar_pose_np = tar_pose.detach().cpu().numpy()
                rec_pose_np = rec_pose.detach().cpu().numpy()
                rec_trans_np = rec_trans.detach().cpu().numpy().reshape(bs*n, 3)
                rec_exp_np = rec_exps.detach().cpu().numpy().reshape(bs*n, 100) 
                tar_exp_np = tar_exps.detach().cpu().numpy().reshape(bs*n, 100)
                tar_trans_np = tar_trans.detach().cpu().numpy().reshape(bs*n, 3)
                gt_npz = np.load(self.args.data_path+self.args.pose_rep +"/"+test_seq_list.iloc[its]['id']+".npz", allow_pickle=True)
                np.savez(results_save_path+"gt_"+test_seq_list.iloc[its]['id']+'.npz',
                    betas=gt_npz["betas"],
                    poses=tar_pose_np,
                    expressions=tar_exp_np,
                    trans=tar_trans_np,
                    model='smplx2020',
                    gender='neutral',
                    mocap_frame_rate = 30 ,
                )
                np.savez(results_save_path+"res_"+test_seq_list.iloc[its]['id']+'.npz',
                    betas=gt_npz["betas"],
                    poses=rec_pose_np,
                    expressions=rec_exp_np,
                    trans=rec_trans_np,
                    model='smplx2020',
                    gender='neutral',
                    mocap_frame_rate = 30,
                )
                total_length += n
                render_vid_path = other_tools_hf.render_one_sequence_no_gt(
                    results_save_path+"res_"+test_seq_list.iloc[its]['id']+'.npz', 
                    # results_save_path+"gt_"+test_seq_list.iloc[its]['id']+'.npz', 
                    results_save_path,
                    self.audio_path,
                    self.args.data_path_1+"smplx_models/",
                    use_matplotlib = False,
                    args = self.args,
                    )

        result = gr.Video(value=render_vid_path, visible=True)
        end_time = time.time() - start_time
        logger.info(f"total inference time: {int(end_time)} s for {int(total_length/self.args.pose_fps)} s motion")
        return result
       
@logger.catch
def syntalker(audio_path,sample_stratege):
    args = config.parse_args()
    if sample_stratege==0:
        args.use_ddim=True
    elif sample_stratege==1:
        args.use_ddim=False
    print(sample_stratege)
    print(args.use_ddim)
    #os.environ['TRANSFORMERS_CACHE'] = args.data_path_1 + "hub/"
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
    # dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    #logger_tools.set_args_and_logger(args, rank)
    other_tools_hf.set_random_seed(args)
    other_tools_hf.print_exp_info(args)

    # return one intance of trainer
    trainer = BaseTrainer(args, ap = audio_path)
    other_tools.load_checkpoints(trainer.model, args.test_ckpt, args.g_name)
    result = trainer.test_demo(999)
    return result

    


examples = [
    ["demo/examples/2_scott_0_1_1.wav"],
    ["demo/examples/2_scott_0_2_2.wav"],
    ["demo/examples/2_scott_0_3_3.wav"],
    ["demo/examples/2_scott_0_4_4.wav"],
    ["demo/examples/2_scott_0_5_5.wav"],
]

demo = gr.Interface(
    syntalker,  # function
    inputs=[
        # gr.File(label="Please upload SMPL-X file with npz format here.", file_types=["npz", "NPZ"]),
        gr.Audio(),
        gr.Radio(choices=["DDIM", "DDPM"], label="Please select a sample strategy", type="index", value="DDIM"),  # 0 for DDIM, 1 for DDPM
        # gr.File(label="Please upload textgrid format file here.", file_types=["TextGrid", "Textgrid", "textgrid"])
    ],  # input type
    outputs=gr.Video(format="mp4", visible=True),
    title='SynTalker: Enabling Synergistic Full-Body Control in Prompt-Based Co-Speech Motion Generation',
    description="1. Upload your audio.  <br/>\
        2. Then, sit back and wait for the rendering to happen! This may take a while (e.g. 1 minutes) <br/>\
        3. After, you can view the videos.  <br/>\
        4. Notice that we use a fix facial animation, our method only produce body motion. <br/>\
        5. Use DDPM sample strategy will generate a better result, while it will take more inference time.  \
            ",
    article="Project links: [SynTalker](https://robinwitch.github.io/SynTalker-Page). <br/>\
             Reference links: [EMAGE](https://pantomatrix.github.io/EMAGE/). ", 
    examples=examples,
)

            
if __name__ == "__main__":
    os.environ["MASTER_ADDR"]='127.0.0.1'
    os.environ["MASTER_PORT"]='8675'
    #os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    demo.launch(share=True)

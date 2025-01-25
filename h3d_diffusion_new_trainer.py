import train_h3d as train
import os
import time
import csv
import sys
import warnings
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import time
import pprint
from loguru import logger
from utils import rotation_conversions as rc
import smplx
from utils import config, logger_tools, other_tools, metric, data_transfer
from dataloaders import data_tools
from optimizers.optim_factory import create_optimizer
from optimizers.scheduler_factory import create_scheduler
from optimizers.loss_factory import get_loss_func
from dataloaders.data_tools import joints_list
import librosa
from diffusion.model_util import create_gaussian_diffusion
from diffusion.resample import create_named_schedule_sampler
import pickle
from models.motionclip import get_model
import clip
from diffusion.cfg_sampler import ClassifierFreeSampleModel_Bodypart, ClassifierFreeSampleModel, TwoClassifierFreeSampleModel,TwoClassifierFreeSampleModel_Bodypart
sys.path.append('./models')
from models.temos.motionencoder.actor import ActorAgnosticEncoder
from models.temos.textencoder.distillbert_actor import DistilbertActorAgnosticEncoder
from utils.plot_script import plot_3d_motion,smpl_kinematic_chain,recover_from_ric
from models.vq.model import RVQVAE
from moviepy.editor import VideoFileClip, AudioFileClip
from utils.t2m_eval_tools import EvaluatorMDMWrapper,evaluate_fid, evaluate_diversity, evaluate_multimodality, get_metric_statistics, evaluate_matching_score
from dataloaders.h3d_eval_gen import CompMDMGeneratedDataset
from torch.utils.data._utils.collate import default_collate
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from collections import OrderedDict

class CustomTrainer(train.BaseTrainer):
    '''
    Multi-Modal AutoEncoder
    '''
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        # self.joints = self.train_data.joints
        self.ori_joint_list = joints_list[self.args.ori_joints]
        self.tar_joint_list_face = joints_list["beat_smplx_face"]
        self.tar_joint_list_upper = joints_list["beat_smplx_upper"]
        self.tar_joint_list_hands = joints_list["beat_smplx_hands"]
        self.tar_joint_list_lower = joints_list["beat_smplx_lower_no_hips"]
        
        self.upper_joint_num = len(self.tar_joint_list_upper)
        self.hands_joint_num = len(self.tar_joint_list_hands)
        self.lower_joint_num = len(self.tar_joint_list_lower)
       
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
                            
        dim_pose = 156  
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

        dim_pose = 360
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

        dim_pose = 107
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


        self.args.vae_length = 256
        self.args.vae_test_dim = 61
        self.args.vae_layer = 4
        self.args.vae_test_dim = 330
        self.args.vae_layer = 4
        self.args.vae_length = 240


        modelpath = './ckpt/distilbert-base-uncased'
        tmr_base_path = './ckpt/beatx_1-30_amass_h3d_tmr'
        textencoder = DistilbertActorAgnosticEncoder(modelpath, num_layers=4).eval().to(self.rank)
        textencoder.load_state_dict(torch.load(f'{tmr_base_path}/text_epoch=299.ckpt'))
        self.textencoder = textencoder

        motionencoder = ActorAgnosticEncoder(nfeats=623, vae = True, num_layers=4).eval().to(self.rank)
        motionencoder.load_state_dict(torch.load(f'{tmr_base_path}/motion_epoch=299.ckpt'))
        self.motionclip = motionencoder
        
        
        self.vq_model_face.eval()
        self.vq_model_upper.eval()
        self.vq_model_hands.eval()
        self.vq_model_lower.eval()

        self.cls_loss = nn.NLLLoss().to(self.rank)
        self.reclatent_loss = nn.MSELoss().to(self.rank)
        self.vel_loss = torch.nn.L1Loss(reduction='mean').to(self.rank)
        self.rec_loss = get_loss_func("GeodesicLoss").to(self.rank) 
        self.log_softmax = nn.LogSoftmax(dim=2).to(self.rank)
        
        self.diffusion = create_gaussian_diffusion()
        self.schedule_sampler_type = 'uniform'
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, self.diffusion)
        
        joints = [3,6,9,12,13,14,15,16,17,18,19,20,21]
        upper_body_mask = []
        for i in joints:
            upper_body_mask.extend([4+ (i-1)*3, 4+ (i-1)*3+1, 4+ (i-1)*3+2])
            upper_body_mask.extend([4+ 51*3 +(i-1)*6, 4+ 51*3 + (i-1)*6+1, 4+ 51*3 + (i-1)*6+2, 4+ 51*3 + (i-1)*6+3, 4+ 51*3 + (i-1)*6+4, 4+ 51*3 + (i-1)*6+5])
            upper_body_mask.extend([4+ 51*9 +i*3, 4+ 51*9 + i*3+1, 4+ 51*9 + i*3+2])
        

        joints = list(range(22,52))
        hands_body_mask = []
        for i in joints:
            hands_body_mask.extend([4+ (i-1)*3, 4+ (i-1)*3+1, 4+ (i-1)*3+2])
            hands_body_mask.extend([4+ 51*3 +(i-1)*6, 4+ 51*3 + (i-1)*6+1, 4+ 51*3 + (i-1)*6+2, 4+ 51*3 + (i-1)*6+3, 4+ 51*3 + (i-1)*6+4, 4+ 51*3 + (i-1)*6+5])
            hands_body_mask.extend([4+ 51*9 +i*3, 4+ 51*9 + i*3+1, 4+ 51*9 + i*3+2])


        joints = [0,1,2,4,5,7,8,10,11]
        lower_body_mask = list(range(0, 4))+list(range(619,623))
        for i in joints:
            if i>0:
                lower_body_mask.extend([4+ (i-1)*3, 4+ (i-1)*3+1, 4+ (i-1)*3+2])
                lower_body_mask.extend([4+ 51*3 +(i-1)*6, 4+ 51*3 + (i-1)*6+1, 4+ 51*3 + (i-1)*6+2, 4+ 51*3 + (i-1)*6+3, 4+ 51*3 + (i-1)*6+4, 4+ 51*3 + (i-1)*6+5])
            lower_body_mask.extend([4+ 51*9 +i*3, 4+ 51*9 + i*3+1, 4+ 51*9 + i*3+2])

        
        self.joint_mask_upper = upper_body_mask
        self.joint_mask_hands = hands_body_mask
        self.joint_mask_lower = lower_body_mask
        self.test_prompt_list = \
        [
            "A person is walking forward",
            "A man is kneel down",
            "A man mimics likes zombies",
            "A person raises up left hand",
            "A person raises up right hand",
            "A person stands on left foot",
            "A person stands on right foot",
            "A person holds a cup of tea in left hand",
            "A person sides lunge forward",
            "A person sits and holds a cup of tea in the left hand",
            "A person sits and holds a cup of tea in the right hand",
            "A person sits and raise up right hands",
            "A person sits and raise up left hands",
        ]
        
        
        self.test_prompt_body_part_list = \
        [
            {
                "upper_mask":None,
                "hands_mask":None,
                "lower_mask":"A person is walking forward",
            },
            {
                "upper_mask":None,
                "hands_mask":None,
                "lower_mask":"A man is kneel down",
            },
            {
                "upper_mask":"A man mimics likes zombies",
                "hands_mask":None,
                "lower_mask":None,
            },
            {
                "upper_mask":"A person raises up left hand",
                "hands_mask":None,
                "lower_mask":None,
            },
            {
                "upper_mask":"A person raises up right hand",
                "hands_mask":None,
                "lower_mask":None,
            },    
            {
                "upper_mask":None,
                "hands_mask":None,
                "lower_mask":"A person stands on left foot",
            },
            {
                "upper_mask":None,
                "hands_mask":None,
                "lower_mask":"A person stands on right foot",
            },        
            {
                "upper_mask":"A person holds a cup of tea in left hand",
                "hands_mask":None,
                "lower_mask":None,
            },
            {
                "upper_mask":None,
                "hands_mask":None,
                "lower_mask":"A person sides lunge forward",
            },
            {
                "upper_mask":"A person holds a cup of tea in the left hand",
                "hands_mask":None,
                "lower_mask":"A person sits",
            },
            {
                "upper_mask":"A person holds a cup of tea in the right hand",
                "hands_mask":None,
                "lower_mask":"A person sits",
            },    
            {
                "upper_mask":"A person raises up right hands",
                "hands_mask":None,
                "lower_mask":"A person sits",
            },    
            {
                "upper_mask":"A person raises up left hands",
                "hands_mask":None,
                "lower_mask":"A person sits",
            },       
        ]
        self.test_prompt_dict = dict(zip(self.test_prompt_list, self.test_prompt_body_part_list))

        # [
        #     "walking forward",
        #     "a person is lying down",
        #     "a person is sitting",
        #     "a person walks in a counterclockwise circle",
        #     "a person walks in a clockwise circle",
        #     "a person walks and then makes a right turn",
        #     "a man raise right arm",
        #     "a person walks forward and stumbles",
        #     "a man takes a large quick step to his left then stops",
        #     "the person is walking a clockwise circle",
        #     "the person is walking a counter-clockwise circle",
        #     "a person walks forward and suddenly turn around",
        #     "a man is playing the guitar",
        #     "person walks up and squats slightly to pose a position",
        #     "a person squats down then jumps",
        #     "a person swings their arm as they jump",
        # ]
        


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
       

    def train_load_data(self, dict_data):
        tar_pose_raw = dict_data["pose"]
        tar_pose = tar_pose_raw.to(self.rank)
        
        tmr_tar_pose = dict_data["tmr_tar_pose"]
        tmr_tar_pose = tmr_tar_pose.to(self.rank)
        
        
        in_audio = dict_data["audio"].to(self.rank) 
        in_word = dict_data["word"].to(self.rank)
        tar_id = dict_data["id"][:,0,0].to(self.rank).long()
        prompt_text = dict_data["prompt_text"]
        bs, n, j = tar_pose.shape[0], tar_pose.shape[1], self.joints

        tar_pose_upper = tar_pose[..., self.joint_mask_upper]
        tar_pose_hands = tar_pose[..., self.joint_mask_hands]
        tar_pose_lower = tar_pose[..., self.joint_mask_lower]
        


        latent_upper_top = self.vq_model_upper.map2latent(tar_pose_upper) # bs*n/4
        latent_hands_top = self.vq_model_hands.map2latent(tar_pose_hands) # bs*n/4
        latent_lower_top = self.vq_model_lower.map2latent(tar_pose_lower) # bs*n/4
        
        latent_in = torch.cat([latent_upper_top, latent_hands_top, latent_lower_top], dim=2) / self.vqvae_latent_scale
        
        style_feature = self.motionclip(tmr_tar_pose).loc
        text_feature = None
        if self.args.text_sample_stride > 0:
            use_text_id = torch.where(tar_id == 99)[0][::self.args.text_sample_stride]
            text_feature = self.textencoder(prompt_text).loc
            style_feature[use_text_id] = text_feature[use_text_id]
        # print(tar_index_value_upper_top.shape, index_in.shape)
        return {
            "tar_pose_upper": tar_pose_upper,
            "tar_pose_lower": tar_pose_lower,
            "tar_pose_hands": tar_pose_hands,
            "in_audio": in_audio,
            "in_word": in_word,
            "tar_pose": tar_pose,
            "latent_upper_top": latent_upper_top,
            "latent_hands_top": latent_hands_top,
            "latent_lower_top": latent_lower_top,
            "latent_in":  latent_in,
            "tar_id": tar_id,
            "style_feature":style_feature,
            "text_feature":text_feature,
            "prompt_text":prompt_text,
            "length":dict_data["m_length"],
        }


    def _load_data(self, dict_data):
        tar_pose_raw = dict_data["pose"]
        tar_pose = tar_pose_raw.to(self.rank)
        in_audio = dict_data["audio"].to(self.rank) 
        in_word = dict_data["word"].to(self.rank)
        tar_id = dict_data["id"][:,0,0].to(self.rank).long()
        prompt_text = dict_data["prompt_text"]
        bs, n, j = tar_pose.shape[0], tar_pose.shape[1], self.joints

        tar_pose_upper = tar_pose[..., self.joint_mask_upper]
        tar_pose_hands = tar_pose[..., self.joint_mask_hands]
        tar_pose_lower = tar_pose[..., self.joint_mask_lower]
        


        latent_upper_top = self.vq_model_upper.map2latent(tar_pose_upper) # bs*n/4
        latent_hands_top = self.vq_model_hands.map2latent(tar_pose_hands) # bs*n/4
        latent_lower_top = self.vq_model_lower.map2latent(tar_pose_lower) # bs*n/4
        
        latent_in = torch.cat([latent_upper_top, latent_hands_top, latent_lower_top], dim=2) / self.vqvae_latent_scale
        
        style_feature = self.motionclip(tar_pose).loc
        text_feature = self.textencoder(prompt_text).loc
        if self.args.text_sample_stride > 0:
            use_text_id = torch.where(tar_id == 99)[0][::self.args.text_sample_stride]
            text_feature = self.textencoder(prompt_text).loc
            style_feature[use_text_id] = text_feature[use_text_id]
        # print(tar_index_value_upper_top.shape, index_in.shape)
        return {
            "tar_pose_upper": tar_pose_upper,
            "tar_pose_lower": tar_pose_lower,
            "tar_pose_hands": tar_pose_hands,
            "in_audio": in_audio,
            "in_word": in_word,
            "tar_pose": tar_pose,
            "latent_upper_top": latent_upper_top,
            "latent_hands_top": latent_hands_top,
            "latent_lower_top": latent_lower_top,
            "latent_in":  latent_in,
            "tar_id": tar_id,
            "style_feature":style_feature,
            "text_feature":text_feature,
            "prompt_text":prompt_text,
            "length":dict_data["m_length"],
            "tokens":dict_data["tokens"],
        }
    
    def _g_training(self, loaded_data, use_adv, mode="train", epoch=0):
        bs, n, j = loaded_data["tar_pose"].shape[0], loaded_data["tar_pose"].shape[1], self.joints 
            
        cond_ = {'y':{}}
        cond_['y']['audio'] = loaded_data['in_audio']
        cond_['y']['word'] = loaded_data['in_word']
        cond_['y']['id'] = loaded_data['tar_id']
        cond_['y']['seed'] = loaded_data['latent_in'][:,:self.args.pre_frames]
        cond_['y']['mask'] = (torch.zeros([self.args.batch_size, 1, 1, self.args.pose_length//4]) < 1).cuda()
        cond_['y']['style_feature'] = loaded_data['style_feature']
        x0 = loaded_data['latent_in']
        x0 = x0.permute(0, 2, 1).unsqueeze(2)
        t, weights = self.schedule_sampler.sample(x0.shape[0], x0.device)
        g_loss_final = self.diffusion.training_losses(self.model,x0,t,model_kwargs = cond_)["loss"].mean()
        self.tracker.update_meter("predict_x0_loss", "train", g_loss_final.item())

        if mode == 'train':
            return g_loss_final

    def _g_test(self, loaded_data):
        
        sample_fn = self.diffusion.p_sample_loop
        use_ddim = True
        if use_ddim:
            self.diffusion = create_gaussian_diffusion(use_ddim = use_ddim)
            sample_fn = self.diffusion.ddim_sample_loop
        
        mode = 'test'
        bs, n, j = loaded_data["tar_pose"].shape[0], loaded_data["tar_pose"].shape[1], self.joints 
        tar_pose = loaded_data["tar_pose"]
        in_word = loaded_data["in_word"]
        in_audio = loaded_data["in_audio"]
        in_x0 = loaded_data['latent_in']
        in_seed = loaded_data['latent_in']
        
        cfg_scale = 1
        cfg_scale_prompt = self.args.prompt_scale
        cfg_scale_audio = self.args.audio_scale
        if isinstance(self.model,ClassifierFreeSampleModel) or isinstance(self.model,TwoClassifierFreeSampleModel):
            text_prompt = self.prompt
            #text_prompt = "stand in left foot"
            style_feature = self.textencoder(text_prompt).loc.cuda()
        
        if isinstance(self.model,ClassifierFreeSampleModel_Bodypart) or isinstance(self.model,TwoClassifierFreeSampleModel_Bodypart):
            
            auto = False
            if auto:
                print("execute this")
                prompt = self.test_prompt_dict[self.prompt].copy()
            else:
                upper_prompt = self.args.upper_prompt
                hands_prompt = self.args.hands_prompt
                lower_prompt = self.args.lower_prompt
                
                #lower_prompt = "a person walks forward"
                prompt = {
                            "upper_mask":upper_prompt,
                            "hands_mask":hands_prompt,
                            "lower_mask":lower_prompt,
                            }
            if isinstance(prompt,dict):
                for key , value in prompt.items():
                    if value is None: continue
                    #clip_token = clip.tokenize(value).cuda()
                    prompt[key] = self.textencoder(value).loc.cuda()
                style_feature = prompt

        remain = n%8
        if remain != 0:
            tar_pose = tar_pose[:, :-remain, :]
            in_word = in_word[:, :-remain]
            in_x0 = in_x0[:, :-remain, :]
            in_seed = in_seed[:, :-remain, :]
            n = n - remain

        
        rec_all_face = []
        rec_all_upper = []
        rec_all_lower = []
        rec_all_hands = []

        vqvae_squeeze_scale = 4
        roundt = (n - self.args.pre_frames * vqvae_squeeze_scale) // (self.args.pose_length - self.args.pre_frames * vqvae_squeeze_scale)
        remain = (n - self.args.pre_frames * vqvae_squeeze_scale) % (self.args.pose_length - self.args.pre_frames * vqvae_squeeze_scale)
        round_l = self.args.pose_length - self.args.pre_frames * vqvae_squeeze_scale
        
        for i in range(0, roundt):
            in_word_tmp = in_word[:, i*(round_l):(i+1)*(round_l)+self.args.pre_frames * vqvae_squeeze_scale]
            # audio fps is 16000 and pose fps is 30
            in_audio_tmp = in_audio[:, i*(16000//30*round_l):(i+1)*(16000//30*round_l)+16000//30*self.args.pre_frames * vqvae_squeeze_scale]
            in_id_tmp = loaded_data['tar_id']#[:, i*(round_l):(i+1)*(round_l)+self.args.pre_frames]
            in_seed_tmp = in_seed[:, i*(round_l):(i+1)*(round_l)+self.args.pre_frames]
            mask_val = torch.ones(bs, self.args.pose_length, self.args.pose_dims+3+4).float().cuda()
            mask_val[:, :self.args.pre_frames, :] = 0.0
            if i == 0:
                in_seed_tmp = in_seed[:, i*(round_l//vqvae_squeeze_scale):(i+1)*(round_l//vqvae_squeeze_scale)+self.args.pre_frames, :]
            else:
                in_seed_tmp = in_seed[:, i*(round_l//vqvae_squeeze_scale):(i+1)*(round_l//vqvae_squeeze_scale)+self.args.pre_frames, :]
                # print(latent_all_tmp.shape, latent_last.shape)
                in_seed_tmp[:, :self.args.pre_frames, :] = latent_last[:, -self.args.pre_frames:, :]
            
            
            cond_ = {'y':{}}
            cond_['y']['audio'] = in_audio_tmp
            cond_['y']['word'] = in_word_tmp
            cond_['y']['id'] = in_id_tmp
            cond_['y']['seed'] =in_seed_tmp[:,:self.args.pre_frames]
            cond_['y']['mask'] = (torch.zeros([self.args.batch_size, 1, 1, self.args.pose_length]) < 1).cuda()
            cond_['y']['style_feature'] = style_feature
            if isinstance(self.model,TwoClassifierFreeSampleModel):
                cond_['y']['scale_prompt'] = torch.ones(1).cuda() * cfg_scale_prompt
                cond_['y']['scale_audio'] = torch.ones(1).cuda() * cfg_scale_audio
            else:
                cond_['y']['scale'] = torch.ones(1).cuda() * cfg_scale_prompt
            shape_ = (1, 1536, 1, 32)
            sample = sample_fn(
                self.model,
                shape_,
                clip_denoised=False,
                model_kwargs=cond_,
                skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                init_image=None,
                progress=True,
                dump_steps=None,
                noise=None,  # None, torch.randn(*shape_, device=mydevice)
                const_noise=False,
            )
            sample = sample.squeeze().permute(1,0).unsqueeze(0)
            latent_last = sample.clone()
            
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



        n = n - remain
        tar_pose = tar_pose[:, :n, :]

        rec_pose = torch.zeros_like(tar_pose)
        rec_pose[..., self.joint_mask_upper] = rec_upper
        rec_pose[..., self.joint_mask_hands] = rec_hands
        rec_pose[..., self.joint_mask_lower] = rec_lower

        tar_audio = in_audio[:, :int(n/30*16000)]
        
        return {
            'rec_pose': rec_pose,
            'tar_pose': tar_pose,
            'tar_audio':tar_audio,            
        }
    
    
    
    def _g_val(self, loaded_data):
        
        #sample_fn = self.diffusion.ddim_sample_loop
        sample_fn = self.diffusion.p_sample_loop
        
        mode = 'test'
        bs, n, j = loaded_data["tar_pose"].shape[0], loaded_data["tar_pose"].shape[1], self.joints 
        tar_pose = loaded_data["tar_pose"]
        in_word = loaded_data["in_word"]
        in_audio = loaded_data["in_audio"]
        in_x0 = loaded_data['latent_in']
        in_seed = loaded_data['latent_in']
                
        
        
        cfg_scale = 1
        cfg_scale_prompt = self.args.prompt_scale
        cfg_scale_audio = 0#self.args.audio_scale
        if isinstance(self.model,ClassifierFreeSampleModel) or isinstance(self.model,TwoClassifierFreeSampleModel):
            if hasattr(self,'prompt'):
                text_prompt = self.prompt
                style_feature = self.textencoder(text_prompt).loc.cuda()
            else:
                style_feature = loaded_data['text_feature']    
        if isinstance(self.model,ClassifierFreeSampleModel_Bodypart):
            prompt = {
                        "upper_mask":None,
                        "hands_mask":None,
                        "lower_mask":"a person passes something to the right",
                        }
            if isinstance(prompt,dict):
                for key , value in prompt.items():
                    if value is None: continue
                    #clip_token = clip.tokenize(value).cuda()
                    prompt[key] = self.textencoder(value).loc.cuda()
                style_feature = prompt

        #这里强行把长度扩大3倍,为了最后能得到一个296长度的输出
        repeat_time = 3 
        tar_pose = tar_pose.repeat(1, repeat_time, 1)
        in_word = in_word.repeat(1, repeat_time)
        in_x0 = tar_pose.repeat(1, repeat_time, 1)
        #in_seed = tar_pose.repeat(1, repeat_time, 1)
        in_audio = in_audio.repeat(1, repeat_time, 1)
        n=n*repeat_time
        
        remain = n%8
        if remain != 0:
            tar_pose = tar_pose[:, :-remain, :]
            in_word = in_word[:, :-remain]
            in_x0 = in_x0[:, :-remain, :]
            in_seed = in_seed[:, :-remain, :]
            n = n - remain

        
        rec_all_face = []
        rec_all_upper = []
        rec_all_lower = []
        rec_all_hands = []
        # rec_index_all_face_bot = []
        # rec_index_all_upper_bot = []
        # rec_index_all_lower_bot = []
        # rec_index_all_hands_bot = []
        
        vqvae_squeeze_scale = 4
        roundt = (n - self.args.pre_frames * vqvae_squeeze_scale) // (self.args.pose_length - self.args.pre_frames * vqvae_squeeze_scale)
        remain = (n - self.args.pre_frames * vqvae_squeeze_scale) % (self.args.pose_length - self.args.pre_frames * vqvae_squeeze_scale)
        round_l = self.args.pose_length - self.args.pre_frames * vqvae_squeeze_scale
        
        # pad latent_all_9 to the same length with latent_all
        # if n - latent_all_9.shape[1] >= 0:
        #     latent_all = torch.cat([latent_all_9, torch.zeros(bs, n - latent_all_9.shape[1], latent_all_9.shape[2]).cuda()], dim=1)
        # else:
        #     latent_all = latent_all_9[:, :n, :]

        for i in range(0, roundt):
            in_word_tmp = in_word[:, i*(round_l):(i+1)*(round_l)+self.args.pre_frames * vqvae_squeeze_scale]
            # audio fps is 16000 and pose fps is 30
            in_audio_tmp = in_audio[:, i*(16000//30*round_l):(i+1)*(16000//30*round_l)+16000//30*self.args.pre_frames * vqvae_squeeze_scale]
            in_id_tmp = loaded_data['tar_id']#[:, i*(round_l):(i+1)*(round_l)+self.args.pre_frames]
            in_seed_tmp = in_seed[:, i*(round_l):(i+1)*(round_l)+self.args.pre_frames]
            mask_val = torch.ones(bs, self.args.pose_length, self.args.pose_dims+3+4).float().cuda()
            mask_val[:, :self.args.pre_frames, :] = 0.0
            if i == 0:
                in_seed_tmp = in_seed[:, i*(round_l//vqvae_squeeze_scale):(i+1)*(round_l//vqvae_squeeze_scale)+self.args.pre_frames, :]
            else:
                # in_seed_tmp = in_seed[:, i*(round_l//vqvae_squeeze_scale):(i+1)*(round_l//vqvae_squeeze_scale)+self.args.pre_frames, :]
                # print(latent_all_tmp.shape, latent_last.shape)
                in_seed_tmp= latent_last[:, -self.args.pre_frames:, :]
            
            
            cond_ = {'y':{}}
            cond_['y']['audio'] = in_audio_tmp
            cond_['y']['word'] = in_word_tmp
            cond_['y']['id'] = in_id_tmp
            cond_['y']['seed'] =in_seed_tmp[:,:self.args.pre_frames]
            cond_['y']['mask'] = (torch.zeros([self.args.batch_size, 1, 1, self.args.pose_length]) < 1).cuda()
            cond_['y']['style_feature'] = style_feature
            if isinstance(self.model,TwoClassifierFreeSampleModel):
                cond_['y']['scale_prompt'] = torch.ones(1).cuda() * cfg_scale_prompt
                cond_['y']['scale_audio'] = torch.ones(1).cuda() * cfg_scale_audio
            else:
                cond_['y']['scale'] = torch.ones(1).cuda() * cfg_scale_prompt
            shape_ = (bs, 1536, 1, 32)
            sample = sample_fn(
                self.model,
                shape_,
                clip_denoised=False,
                model_kwargs=cond_,
                skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                init_image=None,
                progress=True,
                dump_steps=None,
                noise=None,  # None, torch.randn(*shape_, device=mydevice)
                const_noise=False,
            )
            sample = sample.squeeze().permute(0,2,1)
            latent_last = sample.clone()
            
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



        n = n - remain
        tar_pose = tar_pose[:, :n, :]

        rec_pose = torch.zeros_like(tar_pose)
        rec_pose[..., self.joint_mask_upper] = rec_upper
        rec_pose[..., self.joint_mask_hands] = rec_hands
        rec_pose[..., self.joint_mask_lower] = rec_lower

        tar_audio = in_audio[:, :int(n/30*16000)]
        loaded_data['tokens'] = [t.split('_') for t in loaded_data['tokens']]
        return {
            'rec_pose': rec_pose,
            'tar_pose': tar_pose,
            'tar_audio':tar_audio,
            'prompt_text':loaded_data['prompt_text'],
            'lengths':loaded_data['length'],
            'tokens':loaded_data['tokens'],
        }
    
    


    def train(self, epoch):
        #torch.autograd.set_detect_anomaly(True)
        use_adv = bool(epoch>=self.args.no_adv_epoch)
        self.model.train()
        # self.d_model.train()
        t_start = time.time()
        self.tracker.reset()
        for its, batch_data in enumerate(self.train_loader):
            loaded_data = self.train_load_data(batch_data)
            t_data = time.time() - t_start
    
            self.opt.zero_grad()
            g_loss_final = 0
            g_loss_final += self._g_training(loaded_data, use_adv, 'train', epoch)
            #with torch.autograd.detect_anomaly():
            g_loss_final.backward()
            if self.args.grad_norm != 0: 
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_norm)
            self.opt.step()
            
            mem_cost = torch.cuda.memory_cached() / 1E9
            lr_g = self.opt.param_groups[0]['lr']
            # lr_d = self.opt_d.param_groups[0]['lr']
            t_train = time.time() - t_start - t_data
            t_start = time.time()
            if its % self.args.log_period == 0:
                self.train_recording(epoch, its, t_data, t_train, mem_cost, lr_g)   
            if self.args.debug:
                if its == 1: break
        self.opt_s.step(epoch)
        # self.opt_d_s.step(epoch) 
    

    def test(self, epoch):
        
        results_save_path = self.checkpoint_path + f"/{epoch}/"
        if os.path.exists(results_save_path): 
            return 0
        os.makedirs(results_save_path)
        start_time = time.time()
        total_length = 0
        # it's used for load audio
        test_seq_list = self.test_data.selected_file
        align = 0 
        latent_out = []
        latent_ori = []
        l2_all = 0 
        lvel = 0
        self.model.eval()
        self.model = TwoClassifierFreeSampleModel_Bodypart(self.model)
        #self.model = ClassifierFreeSampleModel_Bodypart(self.model)
        #self.model = TwoClassifierFreeSampleModel(self.model)
        #self.model = ClassifierFreeSampleModel(self.model,self.args.eval)
        fix_root = False
        #self.model = ClassifierFreeSampleModel(self.model,self.args.eval)
        fix_root = True
        test_pure_audio = True
        self.model.eval()
        self.smplx.eval()
        self.eval_copy.eval()
        with torch.no_grad():
        
            for its, batch_data in enumerate(self.test_loader):
                if its>10: break
                loaded_data = self.train_load_data(batch_data)
                self.test_prompt_list = ["prompt"]
                for prompt in self.test_prompt_list:
                    self.prompt = prompt
                    net_out = self._g_test(loaded_data)
                    tar_pose = net_out['tar_pose']
                    rec_pose = net_out['rec_pose']

                    # print(rec_pose.shape, tar_pose.shape)
                    bs, n, j = tar_pose.shape[0], tar_pose.shape[1], self.joints
                    if (30/self.args.pose_fps) != 1:
                        assert 30%self.args.pose_fps == 0
                        n *= int(30/self.args.pose_fps)
                        tar_pose = torch.nn.functional.interpolate(tar_pose.permute(0, 2, 1), scale_factor=30/self.args.pose_fps, mode='linear').permute(0,2,1)
                        rec_pose = torch.nn.functional.interpolate(rec_pose.permute(0, 2, 1), scale_factor=30/self.args.pose_fps, mode='linear').permute(0,2,1)


                    tar_pose = tar_pose.cpu()
                    rec_pose = rec_pose.cpu()
                    total_length += n 
                    # --- save --- #
                    rec_animation_save_path = results_save_path+f"/rec_{its}_{self.prompt}.mp4"
                    skeleton = smpl_kinematic_chain
                    inv_norm = self.args.pose_norm
                    if inv_norm:
                        rec_motion = self.test_loader.dataset.inv_transform(rec_pose[0])
                    else :
                        rec_motion = rec_pose[0]
                    rec_motion = recover_from_ric(rec_motion, 52).numpy()
                    plot_3d_motion(rec_animation_save_path, skeleton, rec_motion, title='None', fps=30)
                    rec_npy_save_path = results_save_path+f"/rec_{its}_{self.prompt}.npy"
                    np.save(rec_npy_save_path, rec_motion)

                    plot_gt = False
                    if plot_gt:
                        tar_animation_save_path = results_save_path+f"/tar_{its}.mp4"
                        skeleton = smpl_kinematic_chain
                        if inv_norm:
                            tar_motion = self.test_loader.dataset.inv_transform(tar_pose[0])
                        else :
                            tar_motion = tar_pose[0]
                        tar_motion = recover_from_ric(tar_motion, 52).numpy()
                        plot_3d_motion(tar_animation_save_path, skeleton, tar_motion, title='None', fps=30)
                    
                    save_audio = True
                    if save_audio:
                        print(test_seq_list.iloc[its]['id'])
                        audio_clip = AudioFileClip(self.args.data_path+"wave16k/"+test_seq_list.iloc[its]['id']+".wav")
                        audio_clip.write_audiofile(results_save_path+f"/rec_{its}_{self.prompt}.mp3")
                    
                    print("mp4 save to: ", rec_animation_save_path)
                    print("npy save to: ", rec_npy_save_path)

        # data_tools.result2target_vis(self.args.pose_version, results_save_path, results_save_path, self.test_demo, False)
        end_time = time.time() - start_time
        logger.info(f"total inference time: {int(end_time)} s for {int(total_length/self.args.pose_fps)} s motion")

    def eval(self, epoch):
        
        results_save_path = self.checkpoint_path + f"/{epoch}/"
        if os.path.exists(results_save_path): 
            return 0
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
        #self.model = TwoClassifierFreeSampleModel(self.model)
        self.model = ClassifierFreeSampleModel(self.model,self.args.eval)
        fix_root = False
        #self.model = ClassifierFreeSampleModel(self.model,self.args.eval)
        #self.model = TwoClassifierFreeSampleModel_Bodypart(self.model)
        fix_root = True
        test_pure_audio = True
        self.model.eval()
        self.smplx.eval()
        self.eval_copy.eval()
        
        def collate_fn(batch):
            batch.sort(key=lambda x: x[3], reverse=True)
            return default_collate(batch)
        
        all_metrics = OrderedDict({'Matching Score': OrderedDict({}),
                                   'R_precision': OrderedDict({}),
                                   'FID': OrderedDict({}),
                                   'Diversity': OrderedDict({}),
                                   })        



        replication_times = 20
        for replication in range(replication_times):
            self.gen_data = __import__(f"dataloaders.h3d_eval_gen", fromlist=["something"]).CustomDataset(self.args, "test")
            self.gt_data = __import__(f"dataloaders.h3d_eval_gt", fromlist=["something"]).CustomDataset(self.args, "test")
            self.gen_loader = torch.utils.data.DataLoader(
                self.gen_data, 
                batch_size=32,  
                shuffle=True,  
                num_workers=self.args.loader_workers,
                drop_last=True,
            )
            
            gt_loader = torch.utils.data.DataLoader(
                self.gt_data, 
                batch_size=32,  
                shuffle=True,  
                num_workers=self.args.loader_workers,
                drop_last=True,
                collate_fn=collate_fn,
            )
            
            with torch.no_grad():
                re = []
                for its, batch_data in enumerate(self.gen_loader):
                    loaded_data = self._load_data(batch_data)
                    net_out = self._g_val(loaded_data)
                    re.append(net_out)
            
            sample_motion = []
            sample_length = []
            sample_caption = []
            sample_tokens = []
            sample_cap_len = []
            
            for i in re:
                sample_motion += [i['rec_pose'][bs_i] for bs_i in range(32)]
                sample_length += [i['lengths'][bs_i] for bs_i in range(32)]
                sample_caption += [i['prompt_text'][bs_i] for bs_i in range(32)]
                sample_tokens += [i['tokens'][bs_i] for bs_i in range(32)]
                sample_cap_len += [len(i['tokens'][bs_i]) for bs_i in range(32)]
            
            sample_gen = []
            for i in range(len(sample_motion)):
                sample_gen.append({
                    'motion': sample_motion[i].squeeze().cpu().numpy(),
                    'length': sample_length[i].cpu().numpy(),
                    'caption': sample_caption[i],
                    'tokens': sample_tokens[i],
                    'cap_len': sample_cap_len[i],
                })
            

            gen_dataset = CompMDMGeneratedDataset(sample_gen)

            motion_loader = DataLoader(gen_dataset, batch_size=32, collate_fn=collate_fn, drop_last=True, num_workers=4)
            
            motion_loaders = {}
            motion_loaders['ground truth'] = gt_loader
            motion_loaders['vald'] = motion_loader
            
            eval_wrapper = EvaluatorMDMWrapper('humanml', 'cuda:0')
            with torch.no_grad():
                f = open('result.txt','a+')
                print(f'==================== Replication {replication} ====================')
                print(f'==================== Replication {replication} ====================', file=f, flush=True)
                mat_score_dict, R_precision_dict, acti_dict = evaluate_matching_score(eval_wrapper, motion_loaders, f)
                fid_score_dict = evaluate_fid(eval_wrapper, gt_loader, acti_dict, f)
                div_score_dict = evaluate_diversity(acti_dict, f, 300)      
                

                for key, item in mat_score_dict.items():
                    if key not in all_metrics['Matching Score']:
                        all_metrics['Matching Score'][key] = [item]
                    else:
                        all_metrics['Matching Score'][key] += [item]

                for key, item in R_precision_dict.items():
                    if key not in all_metrics['R_precision']:
                        all_metrics['R_precision'][key] = [item]
                    else:
                        all_metrics['R_precision'][key] += [item]

                for key, item in fid_score_dict.items():
                    if key not in all_metrics['FID']:
                        all_metrics['FID'][key] = [item]
                    else:
                        all_metrics['FID'][key] += [item]

                for key, item in div_score_dict.items():
                    if key not in all_metrics['Diversity']:
                        all_metrics['Diversity'][key] = [item]
                    else:
                        all_metrics['Diversity'][key] += [item]
    
                    mean_dict = {}
                
        for metric_name, metric_dict in all_metrics.items():
            print('========== %s Summary ==========' % metric_name)
            print('========== %s Summary ==========' % metric_name, file=f, flush=True)
            for model_name, values in metric_dict.items():
                # print(metric_name, model_name)
                mean, conf_interval = get_metric_statistics(np.array(values), replication_times)
                mean_dict[metric_name + '_' + model_name] = mean
                # print(mean, mean.dtype)
                if isinstance(mean, np.float64) or isinstance(mean, np.float32):
                    print(f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}')
                    print(f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}', file=f, flush=True)
                elif isinstance(mean, np.ndarray):
                    line = f'---> [{model_name}]'
                    for i in range(len(mean)):
                        line += '(top %d) Mean: %.4f CInt: %.4f;' % (i+1, mean[i], conf_interval[i])
                    print(line)
                    print(line, file=f, flush=True)

        end_time = time.time() - start_time
        logger.info(f"total inference time: {int(end_time)} s for {int(total_length/self.args.pose_fps)} s motion")




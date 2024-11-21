# cbh: next to do: change sample to batch sample

import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy

# A wrapper model for Classifier-free guidance **SAMPLING** only
# https://arxiv.org/abs/2207.12598
class ClassifierFreeSampleModel(nn.Module):

    def __init__(self, model,eval = False):
        super().__init__()
        self.model = model  # model is the actual model to run
        self.eval_metric = eval

    def forward(self, x, timesteps, y=None):
        y['uncond_audio'] = True
        out = self.model(x, timesteps, y)
        #return out
        y_uncond = deepcopy(y)
        y_uncond['uncond_audio'] = True
        y_uncond['uncond'] = True
        out_uncond = self.model(x, timesteps, y_uncond)
        if self.eval_metric:
            return out_uncond
        
        return out_uncond + (y['scale'].view(-1, 1, 1, 1) * (out - out_uncond))


class TwoClassifierFreeSampleModel(nn.Module):

    def __init__(self, model,eval = False):
        super().__init__()
        self.model = model  # model is the actual model to run
        self.eval_metric = eval

    def forward(self, x, timesteps, y=None):
        y_uncond = deepcopy(y)
        y_uncond['uncond_audio'] = True
        y_uncond['uncond'] = True
        out_uncond = self.model(x, timesteps, y_uncond)        


        y_uncond_audio = deepcopy(y)
        y_uncond_audio['uncond_audio'] = True
        out_uncond_audio = self.model(x, timesteps, y_uncond_audio)   
        
        
        y_uncond_text = deepcopy(y)
        y_uncond_text['uncond'] = True
        out_uncond_text = self.model(x, timesteps, y_uncond_text)   

        return out_uncond + (y['scale_audio'].view(-1, 1, 1, 1) * (out_uncond_text - out_uncond)) + (y['scale_prompt'].view(-1, 1, 1, 1) * (out_uncond_audio - out_uncond))


class TwoClassifierFreeSampleModel_Bodypart(nn.Module):

    def __init__(self, model,eval=False):
        super().__init__()
        self.model = TwoClassifierFreeSampleModel(model)  # model is the actual model to run
        self.latent_dim = 1536
        self.eval_metric = eval
        self.audio_scale = 1
        self.prompt_scale = 4

    def forward(self, x, timesteps, y=None):

        #y['uncond_audio'] = True
        #assert cond_mode in ['text', 'action']
        y_uncond = deepcopy(y)
        y_uncond['uncond'] = True
        y_uncond['scale_audio'] = torch.ones(1).cuda() * self.audio_scale
        y_uncond['scale_prompt'] = torch.ones(1).cuda() * 0
        
        # below just test for accelerata sample process for evaluating metric in gesture
        if self.eval_metric:
            y_uncond['style_feature'] = y_uncond['style_feature']['lower_mask']
            out_uncond = self.model(x, timesteps, y_uncond)
            return out_uncond
            
        out = torch.zeros_like(x).to(x.device)
        mask_all = ~ torch.ones(self.latent_dim,dtype = torch.bool).to(out.device)
        for key,value in y['style_feature'].items():
            if value is None: 
                y_part = deepcopy(y)
                y_part['style_feature'] = torch.zeros([1,256]).to(x.device)
                y_part['scale_audio'] = torch.ones(1).cuda() * self.audio_scale
                y_part['scale_prompt']= torch.ones(1).cuda() * 0
                out_part = self.model(x, timesteps, y_part)
                mask_part = mask_dict[key]
                tem_mask = ~ torch.ones(self.latent_dim,dtype = torch.bool).to(out.device)
                tem_mask[mask_part]=True
                mask_all[mask_part]=True
                out_part = (out_part.permute(0,3,2,1)*tem_mask).permute(0,3,2,1)
                out = out + out_part
                
            else:
                y_part = deepcopy(y)
                y_part['style_feature'] = y_part['style_feature'][key]
                
                y_part['scale_audio'] =torch.ones(1).cuda() * 0
                y_part['scale_prompt'] =torch.ones(1).cuda() * self.prompt_scale
                if key in 'upper_mask':
                    y_part['scale_audio'] =torch.ones(1).cuda()
                    y_part['scale_prompt'] =torch.ones(1).cuda() * self.prompt_scale
                
                out_part = self.model(x, timesteps, y_part)
                mask_part = mask_dict[key]
                tem_mask = ~ torch.ones(self.latent_dim,dtype = torch.bool).to(out.device)
                tem_mask[mask_part]=True
                mask_all[mask_part]=True
                out_part = (out_part.permute(0,3,2,1)*tem_mask).permute(0,3,2,1)
                out = out + out_part
            
        
        return out







class ClassifierFreeSampleModel_Bodypart(nn.Module):

    def __init__(self, model,eval=False):
        super().__init__()
        self.model = model  # model is the actual model to run
        self.latent_dim = 1536
        self.eval_metric = eval

    def forward(self, x, timesteps, y=None):

        #y['uncond_audio'] = True
        #assert cond_mode in ['text', 'action']
        y_uncond = deepcopy(y)
        y_uncond['uncond'] = True
        
        # below just test for accelerata sample process for evaluating metric in gesture
        if self.eval_metric:
            y_uncond['style_feature'] = y_uncond['style_feature']['lower_mask']
            out_uncond = self.model(x, timesteps, y_uncond)
            return out_uncond
            
        out = torch.zeros_like(x).to(x.device)
        mask_all = ~ torch.ones(self.latent_dim,dtype = torch.bool).to(out.device)
        for key,value in y['style_feature'].items():
            if value is None: continue
            y_part = deepcopy(y)
            y_part['style_feature'] = y_part['style_feature'][key]
            y_part['uncond_audio'] = True
            out_part = self.model(x, timesteps, y_part)
            mask_part = mask_dict[key]
            tem_mask = ~ torch.ones(self.latent_dim,dtype = torch.bool).to(out.device)
            tem_mask[mask_part]=True
            mask_all[mask_part]=True
            out_part = (out_part.permute(0,3,2,1)*tem_mask).permute(0,3,2,1)
            out = out + out_part
            
        y_uncond['style_feature'] = torch.zeros([1,256]).to(x.device)
        out_uncond = self.model(x, timesteps, y_uncond)
        mask_all = ~ mask_all
        
        out = out + (out_uncond.permute(0,3,2,1)*mask_all).permute(0,3,2,1)
        
        return out_uncond + (y['scale'].view(-1, 1, 1, 1) * (out - out_uncond))



mask_dict = {}

# upper_mask = list(range(0,256))
# hands_mask = list(range(256,512))
# lower_mask = list(range(512,768))


upper_mask = list(range(0,512))
hands_mask = list(range(512,1024))
lower_mask = list(range(1024,1536))


mask_dict = {'upper_mask':upper_mask,
             'hands_mask':hands_mask,
             'lower_mask':lower_mask,
             }

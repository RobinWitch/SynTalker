import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils.layer import BasicBlock
from einops import rearrange
import pickle
from .timm_transformer.transformer import Block as mytimmBlock

class MDM(nn.Module):
    def __init__(self, args):
        super().__init__()

    
        njoints=1536 ## in fact is input_size
        nfeats=1
        latent_dim=512
        ff_size=1024
        num_layers=8
        num_heads=4
        dropout=0.1
        ablation=None
        activation="gelu"
        legacy=False
        data_rep='rot6d'
        dataset='amass'
        audio_feat_dim = 64
        emb_trans_dec=False
        audio_rep=''
        n_seed=8
        cond_mode=''
        kargs={}

    
    
        self.input_size = njoints
        self.legacy = legacy
        self.njoints = njoints
        self.nfeats = nfeats
        self.data_rep = data_rep

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation
        self.activation = activation
        self.action_emb = kargs.get('action_emb', None)

        self.input_feats = self.njoints * self.nfeats
        self.use_motionclip = kargs.get('use_motionclip',False)
        
        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.3)
        self.uncon_text_embeddings = nn.Parameter(torch.zeros(1, 256))
        
        
        self.cond_mask_prob_audio = kargs.get('cond_mask_prob_audio', 0)
        self.uncon_audio_embeddings = nn.Parameter(torch.zeros(1, args.audio_f))
        
        
        

        if args.audio_rep == 'onset+amplitude':
            self.WavEncoder = WavEncoder(args.audio_f,audio_in=2)
        self.audio_feat_dim = args.audio_f
        
        self.text_encoder_body = nn.Linear(300, args.audio_f) 
        
        with open(f"{args.data_path}weights/vocab.pkl", 'rb') as f:
            self.lang_model = pickle.load(f)
            pre_trained_embedding = self.lang_model.word_embedding_weights
        self.text_pre_encoder_body = nn.Embedding.from_pretrained(torch.FloatTensor(pre_trained_embedding),freeze=args.t_fix_pre)
        
        
        

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.emb_trans_dec = emb_trans_dec

        self.cond_mode = cond_mode
        self.num_head = 8
        
        self.mytimmblocks = nn.ModuleList([
            mytimmBlock(dim=self.latent_dim,num_heads=self.num_heads,mlp_ratio=self.ff_size//self.latent_dim,drop_path=self.dropout) #hidden是对应于输入x的维度，attn_heads应该是12，这里写1是为了方便调试流程
                for _ in range(self.num_layers)])
            
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
        self.n_seed = n_seed
        

        self.style_dim = 64
        self.embed_style = nn.Linear(6, self.style_dim)
        self.embed_text = nn.Linear(self.input_size*4, self.latent_dim)

            

        self.output_process = OutputProcess(self.data_rep, self.input_feats, self.latent_dim, self.njoints,
                                            self.nfeats)


        self.rel_pos = SinusoidalEmbeddings(self.latent_dim // self.num_head)
        self.input_process = InputProcess(self.data_rep, self.input_feats , self.latent_dim)
        self.input_process2 = nn.Linear(self.latent_dim * 2 + self.audio_feat_dim, self.latent_dim)
        self.input_process3 = nn.Linear(self.latent_dim + 256, self.latent_dim)
        
        self.mix_audio_text = nn.Linear(args.audio_f+args.word_f,256)


            

    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return self.uncon_text_embeddings.repeat([bs,1])
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask) + self.uncon_text_embeddings.repeat([bs,1]) * mask
                
        else:
            return cond
        
        
    def mask_cond_audio(self, cond, force_mask=False):
        bs,s, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob_audio > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob_audio).view(bs, 1,1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond
        
    def mask_cond_text(self, cond, force_mask=False):
        bs,s= cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob_audio > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob_audio).view(bs,1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond

    def forward(self, x, timesteps, y=None,uncond_info=False):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        seed: [batch_size, njoints, nfeats]
        """
        _,_,_,noise_length = x.shape
        y = y.copy()
        
        bs, njoints, nfeats, nframes = x.shape      # 300 ,1141, 1, 88
        emb_t = self.embed_timestep(timesteps)  # [1, bs, d], (1, 2, 256)

        force_mask = y.get('uncond', False)  # False
        force_mask_audio = y.get('uncond_audio',False)
        #force_mask=uncond_info
        # 这个是临时起作用的，仅在推理的时候使用
        #if self.training is False and force_mask is False:
        # y['audio'] = torch.zeros_like(y['audio']).to(y['audio'].device)
        # y['word'] = torch.zeros_like(y['word']).to(y['word'].device)
        
        if self.n_seed != 0:
            embed_text = self.embed_text(y['seed'].reshape(bs, -1))       # (bs, 256-64)
            emb_seed = embed_text

        
        audio_feat = y['audio']
        audio_feat = self.mask_cond_audio(audio_feat,force_mask=force_mask_audio)
        audio_feat = self.WavEncoder(audio_feat).permute(1, 0, 2)
        # audio_feat = self.mask_audio_cond(audio_feat,force_mask=force_mask)
        # 下面这点需要稍微注意一下，音频方面我不想用cfg
        text_feat = y['word']
        text_feat = self.mask_cond_text(text_feat,force_mask=force_mask_audio)
        text_feat = self.text_pre_encoder_body(text_feat)
        text_feat = self.text_encoder_body(text_feat).permute(1, 0, 2)

        at_feat = torch.cat([audio_feat,text_feat],dim=2)
        at_feat = self.mix_audio_text(at_feat)
        at_feat = F.avg_pool1d(at_feat.permute(1,2,0), 4).permute(2,0,1)    #这行的意义是我motion压了4倍
        # This part is test for timm transformer blocks
        x = x.reshape(bs, njoints * nfeats, 1, nframes)  # [300, 1141, 1, 88] -> [300, 1141, 1, 88]
        # self-attention
        x_ = self.input_process(x)  # [300, 1141, 1, 88] -> [88, 300, 256]

        # local-cross-attention

        xseq = torch.cat((x_, at_feat), axis=2)  # [88, 300, 256], [88, 300, 64] -> [88, 300, 320]
        # all frames
        embed_style_2 = (emb_seed + emb_t).repeat(nframes, 1, 1)  # [300, 256] ,[1, 300, 256] -> [88, 300, 256]
        xseq = torch.cat((embed_style_2, xseq), axis=2)  # -> [88, 300, 576]
        xseq = self.input_process2(xseq)    #[88, 300, 576] -> [88, 300, 256]
        
        xseq = torch.cat((xseq, self.mask_cond(y['style_feature'],force_mask).unsqueeze(0).repeat(nframes, 1, 1)), axis = 2)
        xseq = self.input_process3(xseq)
        
        # 下面10行都是位置编码,感觉加了会好一点点，不知道是不是错觉s
        xseq = xseq.permute(1, 0, 2)  # [88, 300, 256] -> [300, 88, 256]
        xseq = xseq.view(bs, nframes, self.num_head, -1) # [300, 88, 256] -> [300, 88, 8, 32]
        xseq = xseq.permute(0, 2, 1, 3)  # [300, 88, 8, 32] -> [300, 8, 88, 32]
        xseq = xseq.reshape(bs * self.num_head, nframes, -1) # [300, 8, 88, 32] -> [2400, 88, 32]
        pos_emb = self.rel_pos(xseq)  # (88, 32)
        xseq, _ = apply_rotary_pos_emb(xseq, xseq, pos_emb) # [2400, 88, 32]
        xseq_rpe = xseq.reshape(bs, self.num_head, nframes, -1) # [300, 8, 88, 32]
        xseq = xseq_rpe.permute(0, 2, 1, 3)  # [300, 8, 88, 32] -> [300, 88, 8, 32]
        xseq = xseq.view(bs, nframes, -1)   # [300, 88, 8, 32] -> [300, 88, 256]
        
        for block in self.mytimmblocks:
            xseq = block(xseq)
        
        xseq = xseq.permute(1, 0, 2)    # [300, 88, 256] -> [88 ,300, 256]
        output = xseq                


        output = self.output_process(output)  # [88, 300, 256] -> [300, 1141, 1, 88]
        return output[...,:noise_length]


    @staticmethod
    def apply_rotary(x, sinusoidal_pos):
        sin, cos = sinusoidal_pos
        x1, x2 = x[..., 0::2], x[..., 1::2]
        # 如果是旋转query key的话，下面这个直接cat就行，因为要进行矩阵乘法，最终会在这个维度求和。（只要保持query和key的最后一个dim的每一个位置对应上就可以）
        # torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
        # 如果是旋转value的话，下面这个stack后再flatten才可以，因为训练好的模型最后一个dim是两两之间交替的。
        return torch.stack([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).flatten(-2, -1)



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)      # (5000, 128)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)     # (5000, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)



class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class InputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        if self.data_rep == 'rot_vel':
            self.velEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        bs, njoints, nfeats, nframes = x.shape
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints*nfeats)

        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            x = self.poseEmbedding(x)  # [seqlen, bs, d]
            return x
        elif self.data_rep == 'rot_vel':
            first_pose = x[[0]]  # [1, bs, 150]
            first_pose = self.poseEmbedding(first_pose)  # [1, bs, d]
            vel = x[1:]  # [seqlen-1, bs, 150]
            vel = self.velEmbedding(vel)  # [seqlen-1, bs, d]
            return torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, d]
        else:
            raise ValueError


class OutputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim, njoints, nfeats):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)
        if self.data_rep == 'rot_vel':
            self.velFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        nframes, bs, d = output.shape
        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            output = self.poseFinal(output)  # [88, 300, 256] -> [88, 300, 1141]
        elif self.data_rep == 'rot_vel':
            first_pose = output[[0]]  # [1, bs, d]
            first_pose = self.poseFinal(first_pose)  # [1, bs, 150]
            vel = output[1:]  # [seqlen-1, bs, d]
            vel = self.velFinal(vel)  # [seqlen-1, bs, 150]
            output = torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, 150]
        else:
            raise ValueError
        output = output.reshape(nframes, bs, self.njoints, self.nfeats)
        output = output.permute(1, 2, 3, 0)  # [bs, njoints, nfeats, nframes]
        return output


class WavEncoder(nn.Module):
    def __init__(self, out_dim, audio_in=1):
        super().__init__() 
        self.out_dim = out_dim
        self.feat_extractor = nn.Sequential( 
                BasicBlock(audio_in, out_dim//4, 15, 5, first_dilation=1700, downsample=True),
                BasicBlock(out_dim//4, out_dim//4, 15, 6, first_dilation=0, downsample=True),
                BasicBlock(out_dim//4, out_dim//4, 15, 1, first_dilation=7, ),
                BasicBlock(out_dim//4, out_dim//2, 15, 6, first_dilation=0, downsample=True),
                BasicBlock(out_dim//2, out_dim//2, 15, 1, first_dilation=7),
                BasicBlock(out_dim//2, out_dim, 15, 3,  first_dilation=0,downsample=True),     
            )
    def forward(self, wav_data):
        if wav_data.dim() == 2:
            wav_data = wav_data.unsqueeze(1) 
        else:
            wav_data = wav_data.transpose(1, 2)
        out = self.feat_extractor(wav_data)
        return out.transpose(1, 2)

class SinusoidalEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x):
        n = x.shape[-2]
        t = torch.arange(n, device = x.device).type_as(self.inv_freq)
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)

def rotate_half(x):
    x = rearrange(x, 'b ... (r d) -> b (...) r d', r = 2)
    x1, x2 = x.unbind(dim = -2)
    return torch.cat((-x2, x1), dim = -1)

def apply_rotary_pos_emb(q, k, freqs):
    q, k = map(lambda t: (t * freqs.cos()) + (rotate_half(t) * freqs.sin()), (q, k))
    return q, k

if __name__ == '__main__':
    '''
    cd ./main/model
    python mdm.py
    '''
    n_frames = 240

    n_seed = 8

    model = MDM(modeltype='', njoints=1140, nfeats=1, cond_mode = 'cross_local_attention5_style1', action_emb='tensor', audio_rep='mfcc',
                arch='mytrans_enc', latent_dim=256, n_seed=n_seed, cond_mask_prob=0.1)

    x = torch.randn(2, 1140, 1, 88)
    t = torch.tensor([12, 85])

    model_kwargs_ = {'y': {}}
    model_kwargs_['y']['mask'] = (torch.zeros([1, 1, 1, n_frames]) < 1)     # [..., n_seed:]
    model_kwargs_['y']['audio'] = torch.randn(2, 88, 13).permute(1, 0, 2)       # [n_seed:, ...]
    model_kwargs_['y']['style'] = torch.randn(2, 6)
    model_kwargs_['y']['mask_local'] = torch.ones(2, 88).bool()
    model_kwargs_['y']['seed'] = x[..., 0:n_seed]
    y = model(x, t, model_kwargs_['y'])
    print(y.shape)

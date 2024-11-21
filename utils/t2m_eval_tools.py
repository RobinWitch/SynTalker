from collections import OrderedDict
import torch
import numpy as np
from scipy import linalg


# (X - X_train)*(X - X_train) = -2X*X_train + X*X + X_train*X_train
def euclidean_distance_matrix(matrix1, matrix2):
    """
        Params:
        -- matrix1: N1 x D
        -- matrix2: N2 x D
        Returns:
        -- dist: N1 x N2
        dist[i, j] == distance(matrix1[i], matrix2[j])
    """
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * np.dot(matrix1, matrix2.T)    # shape (num_test, num_train)
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)    # shape (num_test, 1)
    d3 = np.sum(np.square(matrix2), axis=1)     # shape (num_train, )
    dists = np.sqrt(d1 + d2 + d3)  # broadcasting
    return dists

def calculate_top_k(mat, top_k):
    size = mat.shape[0]
    gt_mat = np.expand_dims(np.arange(size), 1).repeat(size, 1)
    bool_mat = (mat == gt_mat)
    correct_vec = False
    top_k_list = []
    for i in range(top_k):
#         print(correct_vec, bool_mat[:, i])
        correct_vec = (correct_vec | bool_mat[:, i])
        # print(correct_vec)
        top_k_list.append(correct_vec[:, None])
    top_k_mat = np.concatenate(top_k_list, axis=1)
    return top_k_mat



def calculate_activation_statistics(activations):
    """
    Params:
    -- activation: num_samples x dim_feat
    Returns:
    -- mu: dim_feat
    -- sigma: dim_feat x dim_feat
    """
    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov


def calculate_diversity(activation, diversity_times):
    assert len(activation.shape) == 2
    assert activation.shape[0] > diversity_times
    num_samples = activation.shape[0]

    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    dist = linalg.norm(activation[first_indices] - activation[second_indices], axis=1)
    return dist.mean()


def calculate_multimodality(activation, multimodality_times):
    assert len(activation.shape) == 3
    assert activation.shape[1] > multimodality_times
    num_per_sent = activation.shape[1]

    first_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    second_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    dist = linalg.norm(activation[:, first_dices] - activation[:, second_dices], axis=2)
    return dist.mean()



def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative dataset set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative dataset set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)

def evaluate_matching_score(eval_wrapper, motion_loaders, file):
    match_score_dict = OrderedDict({})
    R_precision_dict = OrderedDict({})
    activation_dict = OrderedDict({})
    print('========== Evaluating Matching Score ==========')
    for motion_loader_name, motion_loader in motion_loaders.items():
        all_motion_embeddings = []
        score_list = []
        all_size = 0
        matching_score_sum = 0
        top_k_count = 0
        # print(motion_loader_name)
        with torch.no_grad():
            for idx, batch in enumerate(motion_loader):
                word_embeddings, pos_one_hots, _, sent_lens, motions, m_lens, _ = batch
                text_embeddings, motion_embeddings = eval_wrapper.get_co_embeddings(
                    word_embs=word_embeddings,
                    pos_ohot=pos_one_hots,
                    cap_lens=sent_lens,
                    motions=motions,
                    m_lens=m_lens
                )
                dist_mat = euclidean_distance_matrix(text_embeddings.cpu().numpy(),
                                                     motion_embeddings.cpu().numpy())
                matching_score_sum += dist_mat.trace()

                argsmax = np.argsort(dist_mat, axis=1)
                top_k_mat = calculate_top_k(argsmax, top_k=3)
                top_k_count += top_k_mat.sum(axis=0)

                all_size += text_embeddings.shape[0]

                all_motion_embeddings.append(motion_embeddings.cpu().numpy())

            all_motion_embeddings = np.concatenate(all_motion_embeddings, axis=0)
            matching_score = matching_score_sum / all_size
            R_precision = top_k_count / all_size
            match_score_dict[motion_loader_name] = matching_score
            R_precision_dict[motion_loader_name] = R_precision
            activation_dict[motion_loader_name] = all_motion_embeddings

        print(f'---> [{motion_loader_name}] Matching Score: {matching_score:.4f}')
        print(f'---> [{motion_loader_name}] Matching Score: {matching_score:.4f}', file=file, flush=True)

        line = f'---> [{motion_loader_name}] R_precision: '
        for i in range(len(R_precision)):
            line += '(top %d): %.4f ' % (i+1, R_precision[i])
        print(line)
        print(line, file=file, flush=True)

    return match_score_dict, R_precision_dict, activation_dict



def evaluate_fid(eval_wrapper, groundtruth_loader, activation_dict, file):
    eval_dict = OrderedDict({})
    gt_motion_embeddings = []
    print('========== Evaluating FID ==========')
    with torch.no_grad():
        for idx, batch in enumerate(groundtruth_loader):
            _, _, _, sent_lens, motions, m_lens, _ = batch
            motion_embeddings = eval_wrapper.get_motion_embeddings(
                motions=motions,
                m_lens=m_lens
            )
            gt_motion_embeddings.append(motion_embeddings.cpu().numpy())
    gt_motion_embeddings = np.concatenate(gt_motion_embeddings, axis=0)
    gt_mu, gt_cov = calculate_activation_statistics(gt_motion_embeddings)

    # print(gt_mu)
    for model_name, motion_embeddings in activation_dict.items():
        mu, cov = calculate_activation_statistics(motion_embeddings)
        # print(mu)
        fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
        print(f'---> [{model_name}] FID: {fid:.4f}')
        print(f'---> [{model_name}] FID: {fid:.4f}', file=file, flush=True)
        eval_dict[model_name] = fid
    return eval_dict


def evaluate_diversity(activation_dict, file, diversity_times):
    eval_dict = OrderedDict({})
    print('========== Evaluating Diversity ==========')
    for model_name, motion_embeddings in activation_dict.items():
        diversity = calculate_diversity(motion_embeddings, diversity_times)
        eval_dict[model_name] = diversity
        print(f'---> [{model_name}] Diversity: {diversity:.4f}')
        print(f'---> [{model_name}] Diversity: {diversity:.4f}', file=file, flush=True)
    return eval_dict


def evaluate_multimodality(eval_wrapper, mm_motion_loaders, file, mm_num_times):
    eval_dict = OrderedDict({})
    print('========== Evaluating MultiModality ==========')
    for model_name, mm_motion_loader in mm_motion_loaders.items():
        mm_motion_embeddings = []
        with torch.no_grad():
            for idx, batch in enumerate(mm_motion_loader):
                # (1, mm_replications, dim_pos)
                motions, m_lens = batch
                motion_embedings = eval_wrapper.get_motion_embeddings(motions[0], m_lens[0])
                mm_motion_embeddings.append(motion_embedings.unsqueeze(0))
        if len(mm_motion_embeddings) == 0:
            multimodality = 0
        else:
            mm_motion_embeddings = torch.cat(mm_motion_embeddings, dim=0).cpu().numpy()
            multimodality = calculate_multimodality(mm_motion_embeddings, mm_num_times)
        print(f'---> [{model_name}] Multimodality: {multimodality:.4f}')
        print(f'---> [{model_name}] Multimodality: {multimodality:.4f}', file=file, flush=True)
        eval_dict[model_name] = multimodality
    return eval_dict


def get_metric_statistics(values, replication_times):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval










import torch.nn as nn
import math
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# from networks.layers import *
import torch.nn.functional as F


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    def __init__(self, margin=3.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive


def init_weight(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose1d):
        nn.init.xavier_normal_(m.weight)
        # m.bias.data.fill_(0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def reparameterize(mu, logvar):
    s_var = logvar.mul(0.5).exp_()
    eps = s_var.data.new(s_var.size()).normal_()
    return eps.mul(s_var).add_(mu)


# batch_size, dimension and position
# output: (batch_size, dim)
def positional_encoding(batch_size, dim, pos):
    assert batch_size == pos.shape[0]
    positions_enc = np.array([
        [pos[j] / np.power(10000, (i-i%2)/dim) for i in range(dim)]
        for j in range(batch_size)
    ], dtype=np.float32)
    positions_enc[:, 0::2] = np.sin(positions_enc[:, 0::2])
    positions_enc[:, 1::2] = np.cos(positions_enc[:, 1::2])
    return torch.from_numpy(positions_enc).float()


def get_padding_mask(batch_size, seq_len, cap_lens):
    cap_lens = cap_lens.data.tolist()
    mask_2d = torch.ones((batch_size, seq_len, seq_len), dtype=torch.float32)
    for i, cap_len in enumerate(cap_lens):
        mask_2d[i, :, :cap_len] = 0
    return mask_2d.bool(), 1 - mask_2d[:, :, 0].clone()


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=300):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, pos):
        return self.pe[pos]


class MovementConvEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MovementConvEncoder, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, 4, 2, 1),
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden_size, output_size, 4, 2, 1),
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out_net = nn.Linear(output_size, output_size)
        self.main.apply(init_weight)
        self.out_net.apply(init_weight)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        # print(outputs.shape)
        return self.out_net(outputs)


class MovementConvDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MovementConvDecoder, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose1d(input_size, hidden_size, 4, 2, 1),
            # nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(hidden_size, output_size, 4, 2, 1),
            # nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out_net = nn.Linear(output_size, output_size)

        self.main.apply(init_weight)
        self.out_net.apply(init_weight)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        return self.out_net(outputs)


class TextVAEDecoder(nn.Module):
    def __init__(self, text_size, input_size, output_size, hidden_size, n_layers):
        super(TextVAEDecoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.emb = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True))

        self.z2init = nn.Linear(text_size, hidden_size * n_layers)
        self.gru = nn.ModuleList([nn.GRUCell(hidden_size, hidden_size) for i in range(self.n_layers)])
        self.positional_encoder = PositionalEncoding(hidden_size)


        self.output = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, output_size)
        )

        #
        # self.output = nn.Sequential(
        #     nn.Linear(hidden_size, hidden_size),
        #     nn.LayerNorm(hidden_size),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(hidden_size, output_size-4)
        # )

        # self.contact_net = nn.Sequential(
        #     nn.Linear(output_size-4, 64),
        #     nn.LayerNorm(64),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(64, 4)
        # )

        self.output.apply(init_weight)
        self.emb.apply(init_weight)
        self.z2init.apply(init_weight)
        # self.contact_net.apply(init_weight)

    def get_init_hidden(self, latent):
        hidden = self.z2init(latent)
        hidden = torch.split(hidden, self.hidden_size, dim=-1)
        return list(hidden)

    def forward(self, inputs, last_pred, hidden, p):
        h_in = self.emb(inputs)
        pos_enc = self.positional_encoder(p).to(inputs.device).detach()
        h_in = h_in + pos_enc
        for i in range(self.n_layers):
            # print(h_in.shape)
            hidden[i] = self.gru[i](h_in, hidden[i])
            h_in = hidden[i]
        pose_pred = self.output(h_in)
        # pose_pred = self.output(h_in) + last_pred.detach()
        # contact = self.contact_net(pose_pred)
        # return torch.cat([pose_pred, contact], dim=-1), hidden
        return pose_pred, hidden


class TextDecoder(nn.Module):
    def __init__(self, text_size, input_size, output_size, hidden_size, n_layers):
        super(TextDecoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.emb = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True))

        self.gru = nn.ModuleList([nn.GRUCell(hidden_size, hidden_size) for i in range(self.n_layers)])
        self.z2init = nn.Linear(text_size, hidden_size * n_layers)
        self.positional_encoder = PositionalEncoding(hidden_size)

        self.mu_net = nn.Linear(hidden_size, output_size)
        self.logvar_net = nn.Linear(hidden_size, output_size)

        self.emb.apply(init_weight)
        self.z2init.apply(init_weight)
        self.mu_net.apply(init_weight)
        self.logvar_net.apply(init_weight)

    def get_init_hidden(self, latent):

        hidden = self.z2init(latent)
        hidden = torch.split(hidden, self.hidden_size, dim=-1)

        return list(hidden)

    def forward(self, inputs, hidden, p):
        # print(inputs.shape)
        x_in = self.emb(inputs)
        pos_enc = self.positional_encoder(p).to(inputs.device).detach()
        x_in = x_in + pos_enc

        for i in range(self.n_layers):
            hidden[i] = self.gru[i](x_in, hidden[i])
            h_in = hidden[i]
        mu = self.mu_net(h_in)
        logvar = self.logvar_net(h_in)
        z = reparameterize(mu, logvar)
        return z, mu, logvar, hidden

class AttLayer(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim):
        super(AttLayer, self).__init__()
        self.W_q = nn.Linear(query_dim, value_dim)
        self.W_k = nn.Linear(key_dim, value_dim, bias=False)
        self.W_v = nn.Linear(key_dim, value_dim)

        self.softmax = nn.Softmax(dim=1)
        self.dim = value_dim

        self.W_q.apply(init_weight)
        self.W_k.apply(init_weight)
        self.W_v.apply(init_weight)

    def forward(self, query, key_mat):
        '''
        query (batch, query_dim)
        key (batch, seq_len, key_dim)
        '''
        # print(query.shape)
        query_vec = self.W_q(query).unsqueeze(-1)       # (batch, value_dim, 1)
        val_set = self.W_v(key_mat)                     # (batch, seq_len, value_dim)
        key_set = self.W_k(key_mat)                     # (batch, seq_len, value_dim)

        weights = torch.matmul(key_set, query_vec) / np.sqrt(self.dim)

        co_weights = self.softmax(weights)              # (batch, seq_len, 1)
        values = val_set * co_weights                   # (batch, seq_len, value_dim)
        pred = values.sum(dim=1)                        # (batch, value_dim)
        return pred, co_weights

    def short_cut(self, querys, keys):
        return self.W_q(querys), self.W_k(keys)


class TextEncoderBiGRU(nn.Module):
    def __init__(self, word_size, pos_size, hidden_size, device):
        super(TextEncoderBiGRU, self).__init__()
        self.device = device

        self.pos_emb = nn.Linear(pos_size, word_size)
        self.input_emb = nn.Linear(word_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        # self.linear2 = nn.Linear(hidden_size, output_size)

        self.input_emb.apply(init_weight)
        self.pos_emb.apply(init_weight)
        # self.linear2.apply(init_weight)
        # self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.hidden = nn.Parameter(torch.randn((2, 1, self.hidden_size), requires_grad=True))

    # input(batch_size, seq_len, dim)
    def forward(self, word_embs, pos_onehot, cap_lens):
        num_samples = word_embs.shape[0]

        pos_embs = self.pos_emb(pos_onehot)
        inputs = word_embs + pos_embs
        input_embs = self.input_emb(inputs)
        hidden = self.hidden.repeat(1, num_samples, 1)

        cap_lens = cap_lens.data.tolist()
        emb = pack_padded_sequence(input_embs, cap_lens, batch_first=True)

        gru_seq, gru_last = self.gru(emb, hidden)

        gru_last = torch.cat([gru_last[0], gru_last[1]], dim=-1)
        gru_seq = pad_packed_sequence(gru_seq, batch_first=True)[0]
        forward_seq = gru_seq[..., :self.hidden_size]
        backward_seq = gru_seq[..., self.hidden_size:].clone()

        # Concate the forward and backward word embeddings
        for i, length in enumerate(cap_lens):
            backward_seq[i:i+1, :length] = torch.flip(backward_seq[i:i+1, :length].clone(), dims=[1])
        gru_seq = torch.cat([forward_seq, backward_seq], dim=-1)

        return gru_seq, gru_last


class TextEncoderBiGRUCo(nn.Module):
    def __init__(self, word_size, pos_size, hidden_size, output_size, device):
        super(TextEncoderBiGRUCo, self).__init__()
        self.device = device

        self.pos_emb = nn.Linear(pos_size, word_size)
        self.input_emb = nn.Linear(word_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, output_size)
        )

        self.input_emb.apply(init_weight)
        self.pos_emb.apply(init_weight)
        self.output_net.apply(init_weight)
        # self.linear2.apply(init_weight)
        # self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.hidden = nn.Parameter(torch.randn((2, 1, self.hidden_size), requires_grad=True))

    # input(batch_size, seq_len, dim)
    def forward(self, word_embs, pos_onehot, cap_lens):
        num_samples = word_embs.shape[0]

        pos_embs = self.pos_emb(pos_onehot)
        inputs = word_embs + pos_embs
        input_embs = self.input_emb(inputs)
        hidden = self.hidden.repeat(1, num_samples, 1)

        cap_lens = cap_lens.data.tolist()
        emb = pack_padded_sequence(input_embs, cap_lens, batch_first=True)

        gru_seq, gru_last = self.gru(emb, hidden)

        gru_last = torch.cat([gru_last[0], gru_last[1]], dim=-1)

        return self.output_net(gru_last)


class MotionEncoderBiGRUCo(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device):
        super(MotionEncoderBiGRUCo, self).__init__()
        self.device = device

        self.input_emb = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, output_size)
        )

        self.input_emb.apply(init_weight)
        self.output_net.apply(init_weight)
        self.hidden_size = hidden_size
        self.hidden = nn.Parameter(torch.randn((2, 1, self.hidden_size), requires_grad=True))

    # input(batch_size, seq_len, dim)
    def forward(self, inputs, m_lens):
        num_samples = inputs.shape[0]

        input_embs = self.input_emb(inputs)
        hidden = self.hidden.repeat(1, num_samples, 1)

        cap_lens = m_lens.data.tolist()
        emb = pack_padded_sequence(input_embs, cap_lens, batch_first=True)

        gru_seq, gru_last = self.gru(emb, hidden)

        gru_last = torch.cat([gru_last[0], gru_last[1]], dim=-1)

        return self.output_net(gru_last)


class MotionLenEstimatorBiGRU(nn.Module):
    def __init__(self, word_size, pos_size, hidden_size, output_size):
        super(MotionLenEstimatorBiGRU, self).__init__()

        self.pos_emb = nn.Linear(pos_size, word_size)
        self.input_emb = nn.Linear(word_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        nd = 512
        self.output = nn.Sequential(
            nn.Linear(hidden_size*2, nd),
            nn.LayerNorm(nd),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(nd, nd // 2),
            nn.LayerNorm(nd // 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(nd // 2, nd // 4),
            nn.LayerNorm(nd // 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(nd // 4, output_size)
        )
        # self.linear2 = nn.Linear(hidden_size, output_size)

        self.input_emb.apply(init_weight)
        self.pos_emb.apply(init_weight)
        self.output.apply(init_weight)
        # self.linear2.apply(init_weight)
        # self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.hidden = nn.Parameter(torch.randn((2, 1, self.hidden_size), requires_grad=True))

    # input(batch_size, seq_len, dim)
    def forward(self, word_embs, pos_onehot, cap_lens):
        num_samples = word_embs.shape[0]

        pos_embs = self.pos_emb(pos_onehot)
        inputs = word_embs + pos_embs
        input_embs = self.input_emb(inputs)
        hidden = self.hidden.repeat(1, num_samples, 1)

        cap_lens = cap_lens.data.tolist()
        emb = pack_padded_sequence(input_embs, cap_lens, batch_first=True)

        gru_seq, gru_last = self.gru(emb, hidden)

        gru_last = torch.cat([gru_last[0], gru_last[1]], dim=-1)

        return self.output(gru_last)





POS_enumerator = {
    'VERB': 0,
    'NOUN': 1,
    'DET': 2,
    'ADP': 3,
    'NUM': 4,
    'AUX': 5,
    'PRON': 6,
    'ADJ': 7,
    'ADV': 8,
    'Loc_VIP': 9,
    'Body_VIP': 10,
    'Obj_VIP': 11,
    'Act_VIP': 12,
    'Desc_VIP': 13,
    'OTHER': 14,
}

from os.path import join as pjoin

def build_models(opt):
    movement_enc = MovementConvEncoder(opt.dim_pose-4, opt.dim_movement_enc_hidden, opt.dim_movement_latent)
    text_enc = TextEncoderBiGRUCo(word_size=opt.dim_word,
                                  pos_size=opt.dim_pos_ohot,
                                  hidden_size=opt.dim_text_hidden,
                                  output_size=opt.dim_coemb_hidden,
                                  device=opt.device)

    motion_enc = MotionEncoderBiGRUCo(input_size=opt.dim_movement_latent,
                                      hidden_size=opt.dim_motion_hidden,
                                      output_size=opt.dim_coemb_hidden,
                                      device=opt.device)

    checkpoint = torch.load(pjoin(opt.checkpoints_dir, opt.dataset_name, 'text_mot_match', 'model', 'finest.tar'),
                            map_location=opt.device)
    movement_enc.load_state_dict(checkpoint['movement_encoder'])
    text_enc.load_state_dict(checkpoint['text_encoder'])
    motion_enc.load_state_dict(checkpoint['motion_encoder'])
    print('Loading Evaluation Model Wrapper (Epoch %d) Completed!!' % (checkpoint['epoch']))
    return text_enc, motion_enc, movement_enc


class EvaluatorModelWrapper(object):

    def __init__(self, opt):

        if opt.dataset_name == 't2m':
            opt.dim_pose = 263
        elif opt.dataset_name == 'kit':
            opt.dim_pose = 251
        else:
            raise KeyError('Dataset not Recognized!!!')

        opt.dim_word = 300
        opt.max_motion_length = 196
        opt.dim_pos_ohot = len(POS_enumerator)
        opt.dim_motion_hidden = 1024
        opt.max_text_len = 20
        opt.dim_text_hidden = 512
        opt.dim_coemb_hidden = 512

        self.text_encoder, self.motion_encoder, self.movement_encoder = build_models(opt)
        self.opt = opt
        self.device = opt.device

        self.text_encoder.to(opt.device)
        self.motion_encoder.to(opt.device)
        self.movement_encoder.to(opt.device)

        self.text_encoder.eval()
        self.motion_encoder.eval()
        self.movement_encoder.eval()

    # Please note that the results does not following the order of inputs
    def get_co_embeddings(self, word_embs, pos_ohot, cap_lens, motions, m_lens):
        with torch.no_grad():
            word_embs = word_embs.detach().to(self.device).float()
            pos_ohot = pos_ohot.detach().to(self.device).float()
            motions = motions.detach().to(self.device).float()

            align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            m_lens = m_lens[align_idx]

            '''Movement Encoding'''
            movements = self.movement_encoder(motions[..., :-4]).detach()
            m_lens = m_lens // self.opt.unit_length
            motion_embedding = self.motion_encoder(movements, m_lens)

            '''Text Encoding'''
            text_embedding = self.text_encoder(word_embs, pos_ohot, cap_lens)
            text_embedding = text_embedding[align_idx]
        return text_embedding, motion_embedding

    # Please note that the results does not following the order of inputs
    def get_motion_embeddings(self, motions, m_lens):
        with torch.no_grad():
            motions = motions.detach().to(self.device).float()

            align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            m_lens = m_lens[align_idx]

            '''Movement Encoding'''
            movements = self.movement_encoder(motions[..., :-4]).detach()
            m_lens = m_lens // self.opt.unit_length
            motion_embedding = self.motion_encoder(movements, m_lens)
        return motion_embedding

# our version
def build_evaluators(opt):
    movement_enc = MovementConvEncoder(opt['dim_pose']-4, opt['dim_movement_enc_hidden'], opt['dim_movement_latent'])
    text_enc = TextEncoderBiGRUCo(word_size=opt['dim_word'],
                                  pos_size=opt['dim_pos_ohot'],
                                  hidden_size=opt['dim_text_hidden'],
                                  output_size=opt['dim_coemb_hidden'],
                                  device=opt['device'])

    motion_enc = MotionEncoderBiGRUCo(input_size=opt['dim_movement_latent'],
                                      hidden_size=opt['dim_motion_hidden'],
                                      output_size=opt['dim_coemb_hidden'],
                                      device=opt['device'])

    ckpt_dir = opt['dataset_name']
    if opt['dataset_name'] == 'humanml':
        ckpt_dir = 't2m'

    checkpoint = torch.load(pjoin(opt['checkpoints_dir'], ckpt_dir, 'text_mot_match', 'model', 'finest.tar'),
                            map_location=opt['device'])
    movement_enc.load_state_dict(checkpoint['movement_encoder'])
    text_enc.load_state_dict(checkpoint['text_encoder'])
    motion_enc.load_state_dict(checkpoint['motion_encoder'])
    print('Loading Evaluation Model Wrapper (Epoch %d) Completed!!' % (checkpoint['epoch']))
    return text_enc, motion_enc, movement_enc

# our wrapper
class EvaluatorMDMWrapper(object):

    def __init__(self, dataset_name, device):
        opt = {
            'dataset_name': dataset_name,
            'device': device,
            'dim_word': 300,
            'max_motion_length': 196,
            'dim_pos_ohot': len(POS_enumerator),
            'dim_motion_hidden': 1024,
            'max_text_len': 20,
            'dim_text_hidden': 512,
            'dim_coemb_hidden': 512,
            'dim_pose': 623 if dataset_name == 'humanml' else 251,
            'dim_movement_enc_hidden': 512,
            'dim_movement_latent': 512,
            'checkpoints_dir': '/mnt/fu09a/chenbohong/gdc/motion-diffusion-model/',
            'unit_length': 4,
        }

        self.text_encoder, self.motion_encoder, self.movement_encoder = build_evaluators(opt)
        self.opt = opt
        self.device = opt['device']

        self.text_encoder.to(opt['device'])
        self.motion_encoder.to(opt['device'])
        self.movement_encoder.to(opt['device'])

        self.text_encoder.eval()
        self.motion_encoder.eval()
        self.movement_encoder.eval()

    # Please note that the results does not following the order of inputs
    def get_co_embeddings(self, word_embs, pos_ohot, cap_lens, motions, m_lens):
        with torch.no_grad():
            word_embs = word_embs.detach().to(self.device).float()
            pos_ohot = pos_ohot.detach().to(self.device).float()
            motions = motions.detach().to(self.device).float()

            align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            m_lens = m_lens[align_idx]

            '''Movement Encoding'''
            movements = self.movement_encoder(motions[..., :-4]).detach()
            m_lens = m_lens // self.opt['unit_length']
            motion_embedding = self.motion_encoder(movements, m_lens)

            '''Text Encoding'''
            text_embedding = self.text_encoder(word_embs, pos_ohot, cap_lens)
            text_embedding = text_embedding[align_idx]
        return text_embedding, motion_embedding

    # Please note that the results does not following the order of inputs
    def get_motion_embeddings(self, motions, m_lens):
        with torch.no_grad():
            motions = motions.detach().to(self.device).float()

            align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            m_lens = m_lens[align_idx]

            '''Movement Encoding'''
            movements = self.movement_encoder(motions[..., :-4]).detach()
            m_lens = m_lens // self.opt['unit_length']
            motion_embedding = self.motion_encoder(movements, m_lens)
        return motion_embedding
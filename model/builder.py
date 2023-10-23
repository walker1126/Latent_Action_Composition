# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
from .backbone_unik import *
from .networks import AutoEncoder2x, AutoEncoder3x

import torch.nn.functional as F
from common import config

#config.initialize()

def get_autoencoder(config):
    
    assert config.name is not None
    print(config.name == 'view')
    if config.name == 'skeleton':
        return AutoEncoder2x(config.mot_en_channels, config.body_en_channels, config.de_channels,
                             global_pool=F.max_pool1d, convpool=nn.MaxPool1d, compress=False)
    elif config.name == 'view':
        return AutoEncoder2x(config.mot_en_channels, config.view_en_channels, config.de_channels,
                             global_pool=F.max_pool1d, convpool=nn.MaxPool1d, compress=False)   #FIXME: max/avg
    else:
        return AutoEncoder3x(config.mot_en_channels, config.body_en_channels,
                             config.view_en_channels, config.de_channels)


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, dim=128, K=65536, m=0.999, T=0.07, mlp=False, 
        num_person=2,
        num_joints=17,
        num_heads=3,
        tau=1,
        in_channels=2):

        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        # new
        config.initializecom()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = UNIK(num_class=dim, num_joints=num_joints, num_person=num_person, tau=tau, num_heads=num_heads, in_channels=in_channels)
        self.encoder_k = UNIK(num_class=dim, num_joints=num_joints, num_person=num_person, tau=tau, num_heads=num_heads, in_channels=in_channels)
        # new
        self.CoM = get_autoencoder(config)
        self.CoM.load_state_dict(torch.load('train_log_COM17/exp_view/model/model_epoch300.pth'))
        self.CoM.eval()
        print('weights loaded')
        for l, module in self.CoM._modules.items():
            print('fixed layers:', l)
            for p in module.parameters():
                p.requires_grad=False




        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K, 30))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = keys#concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size, :] = keys.permute(1,0,2)#keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = x #concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        #torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = 0#torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = x# concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = 0#torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    @torch.no_grad()
    def composition(self, im_q, im_p):
        # new
        N, C, T, V, M = im_q.size()
        #N, C, T, V, M -> N, VC, T
        im_q = im_q.permute(0, 3, 1, 2, 4).contiguous().view(N*M, V*C, T)
        im_p = im_p.permute(0, 3, 1, 2, 4).contiguous().view(N*M, V*C, T)

        im_g = self.CoM.transfer(im_q, im_p)
       # print(im_g.shape)
        im_g = im_g.view(N, M, V, C, T).permute(0, 3, 4, 2, 1).contiguous().detach()
        return im_g

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        N, C, T, V, M = im_q.size()
        # compute query features
        q = self.encoder_q(im_q)  # queries: NxCxT
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys 
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)
            '''
            # new
            #N, C, T, V, M -> N, VC, T
            im_q = im_q.permute(0, 3, 1, 2, 4).contiguous().view(N*M, V*C, T)
            im_g = im_g.permute(0, 3, 1, 2, 4).contiguous().view(N*M, V*C, T)

            im_g = self.CoM.transfer(im_q, im_g)
           # print(im_g.shape)
            im_g = im_g.view(N, M, V, C, T).permute(0, 3, 4, 2, 1).contiguous()

            im_g, idx_unshuffle = self._batch_shuffle_ddp(im_g)
            
            g = self.encoder_k(im_g)  # keys: NxC
            g = nn.functional.normalize(g, dim=1)

            # undo shuffle
            g = self._batch_unshuffle_ddp(g, idx_unshuffle)
            '''
        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos_k = torch.einsum('nct,nct->n', [q, k]).unsqueeze(-1)
        # new
       # l_pos_g = torch.einsum('nc,nc->n', [q, g]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nct,ckt->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos_k, l_neg], dim=1)
        #logits_g = torch.cat([l_pos_g, l_neg], dim=1)

        #logits = torch.cat([logits_k, logits_g], dim=0)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
 
        # labels[1] = 0.7
        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

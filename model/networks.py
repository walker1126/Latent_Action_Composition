import torch
import torch.nn as nn
import torch.nn.functional as F
import random
#from .backbone_unsup_unik import UNIK_Encoder, bn_init

class Encoder(nn.Module):
    def __init__(self, channels, kernel_size=8, global_pool=None, convpool=None, compress=False):
        super(Encoder, self).__init__()

        model = []
        acti = nn.LeakyReLU(0.2)

        nr_layer = len(channels) - 2 if compress else len(channels) - 1

        for i in range(nr_layer):
            if convpool is None:
                pad = (kernel_size - 2) // 2
                model.append(nn.ReflectionPad1d(pad))
                model.append(nn.Conv1d(channels[i], channels[i+1],
                                   kernel_size=kernel_size, stride=2))
                model.append(acti)
            else:
                pad = (kernel_size - 1) // 2
                model.append(nn.ReflectionPad1d(pad))
                model.append(nn.Conv1d(channels[i], channels[i+1],
                                       kernel_size=kernel_size, stride=1))
                model.append(acti)
                model.append(convpool(kernel_size=2, stride=2))

        self.global_pool = global_pool
        self.compress = compress

        self.model = nn.Sequential(*model)
       
        # Action Dictionary
        self.v = nn.Parameter(torch.Tensor(channels[-1], 160))

        nn.init.orthogonal_(self.v)
 
        if self.compress:
            self.conv1x1 = nn.Conv1d(channels[-2], channels[-1], kernel_size=1)

    def forward(self, x):
        #x shape: N VC T
        x = self.model(x)
        if self.global_pool is not None:
            ks = x.shape[-1]
            x = self.global_pool(x, ks)
            if self.compress:
                x = self.conv1x1(x)
        weight = self.v + 1e-8
        Q, R = torch.linalg.qr(weight)  # get eignvector, orthogonal [n1, n2, n3, n4]
        Q_m = Q[:, :128]
        Q_b = Q[:, 128:]

        alp_m = torch.matmul(x.permute(0,2,1), Q_m) / sum(Q_m * Q_m) #  alp: N T 32
        alp_b = torch.matmul(x.permute(0,2,1).mean(1), Q_b) / sum(Q_b * Q_b) # [N,2]
        b = torch.matmul(alp_b, Q_b.permute(1,0)).unsqueeze(2).repeat(1,1,x.shape[-1]) 
        m = torch.matmul(alp_m, Q_m.permute(1,0)).permute(0,2,1) # N, 160, T
        
        return m, b, alp_m, alp_b


class Decoder(nn.Module):
    def __init__(self, channels, kernel_size=7):
        super(Decoder, self).__init__()

        model = []
        pad = (kernel_size - 1) // 2
        acti = nn.LeakyReLU(0.2)

        for i in range(len(channels) - 1):
            model.append(nn.Upsample(scale_factor=2, mode='nearest'))
            model.append(nn.ReflectionPad1d(pad))
            model.append(nn.Conv1d(channels[i], channels[i + 1],
                                            kernel_size=kernel_size, stride=1))
            if i == 0 or i == 1:
                model.append(nn.Dropout(p=0.2))
            if not i == len(channels) - 2:

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class AutoEncoder2x(nn.Module):
    def __init__(self, mot_en_channels, body_en_channels, de_channels, global_pool=None, convpool=None, compress=False):
        super(AutoEncoder2x, self).__init__()
        assert mot_en_channels[0] == de_channels[-1] and \
               mot_en_channels[-1] == de_channels[0]

        self.mot_encoder = Encoder(mot_en_channels)
        self.decoder = Decoder(de_channels)


    def cross(self, x1, x2):
        m1, b1 = self.mot_encoder(x1)
        m2, b2 = self.mot_encoder(x2)

        out1 = self.decoder(m1+b1)
        out2 = self.decoder(m2+b2)
        out12 = self.decoder(m1+b2)
        out21 = self.decoder(m2+b1)

        return out1, out2, out12, out21

    def transfer(self, x1, x2):
        m1, b1, alp_m1, alp_b1 = self.mot_encoder(x1)
        m2, b2, alp_m2, alp_b2 = self.mot_encoder(x2)

        weight = self.mot_encoder.v + 1e-8
        Q, R = torch.linalg.qr(weight)
      
        Q_m = Q[:, :128]
        Q_b = Q[:, 128:]

        # ACtion composition
        alp_m12 = (alp_m1 + alp_m2) / 2          
        m12 = torch.matmul(alp_m12, Q_m.permute(1,0)).permute(0,2,1)
        out12_1 = self.decoder(m12+b1)

        return out12_1

    def cross_with_triplet(self, x1, x2, x12, x21):
        m1, b1, alp_m1, alp_b1 = self.mot_encoder(x1)
        m2, b2, alp_m2, alp_b2 = self.mot_encoder(x2)
       
        out1 = self.decoder(m1+b1)
        out2 = self.decoder(m2+b2)
        out12 = self.decoder(m1+b2)
        out21 = self.decoder(m2+b1)

        m12, b12, alp_m12, alp_b12 = self.mot_encoder(x12)
        m21, b21, alp_m21, alp_b21 = self.mot_encoder(x21)

        outputs = [out1, out2, out12, out21]
        motionvecs = [m1.reshape(m1.shape[0], -1),
                      m2.reshape(m2.shape[0], -1),
                      m12.reshape(m12.shape[0], -1),
                      m21.reshape(m21.shape[0], -1)]
        bodyvecs = [b1.reshape(b1.shape[0], -1),
                      b2.reshape(b2.shape[0], -1),
                      b21.reshape(b21.shape[0], -1),
                      b12.reshape(b12.shape[0], -1)]

        return outputs, motionvecs, bodyvecs

    def forward(self, x):
        m, b = self.mot_encoder(x)
        d = self.decoder(m+b)
        return d


import os
import math
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CBAM(nn.Module):

    def __init__(self, n_channels_in, reduction_ratio, kernel_size):
        super(CBAM, self).__init__()
        self.n_channels_in = n_channels_in
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size

        self.channel_attention = ChannelAttention(n_channels_in, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, f):
        chan_att = self.channel_attention(f)
        # print(chan_att.size())
        fp = chan_att * f
        # print(fp.size())
        spat_att = self.spatial_attention(fp)
        # print(spat_att.size())
        fpp = spat_att * fp
        # print(fpp.size())
        return fpp


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size):
        super(SpatialAttention, self).__init__()
        self.kernel_size = kernel_size

        assert kernel_size % 2 == 1, "Odd kernel size required"
        self.conv = nn.Conv2d(in_channels = 2, out_channels = 1, kernel_size = kernel_size, padding= int((kernel_size-1)/2))
        # batchnorm

    def forward(self, x):
        max_pool = self.agg_channel(x, "max")
        avg_pool = self.agg_channel(x, "avg")
        pool = T.cat([max_pool, avg_pool], dim = 1)
        conv = self.conv(pool)
        conv = conv.repeat(1,x.size()[1],1,1)
        att = T.sigmoid(conv)
        return att

    def agg_channel(self, x, pool = "max"):
        b,c,h,w = x.size()
        x = x.view(b, c, h*w)
        x = x.permute(0,2,1)
        if pool == "max":
            x = F.max_pool1d(x,c)
        elif pool == "avg":
            x = F.avg_pool1d(x,c)
        x = x.permute(0,2,1)
        x = x.view(b,1,h,w)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, n_channels_in, reduction_ratio):
        super(ChannelAttention, self).__init__()
        self.n_channels_in = n_channels_in
        self.reduction_ratio = reduction_ratio
        self.middle_layer_size = int(self.n_channels_in/ float(self.reduction_ratio))

        self.bottleneck = nn.Sequential(
            nn.Linear(self.n_channels_in, self.middle_layer_size),
            nn.ReLU(),
            nn.Linear(self.middle_layer_size, self.n_channels_in)
        )


    def forward(self, x):
        kernel = (x.size()[2], x.size()[3])
        avg_pool = F.avg_pool2d(x, kernel )
        max_pool = F.max_pool2d(x, kernel)


        avg_pool = avg_pool.view(avg_pool.size()[0], -1)
        max_pool = max_pool.view(max_pool.size()[0], -1)


        avg_pool_bck = self.bottleneck(avg_pool)
        max_pool_bck = self.bottleneck(max_pool)

        pool_sum = avg_pool_bck + max_pool_bck

        sig_pool = T.sigmoid(pool_sum)
        sig_pool = sig_pool.unsqueeze(2).unsqueeze(3)

        out = sig_pool.repeat(1,1,kernel[0], kernel[1])
        return out

class CriticNetwork(nn.Module):
    def __init__(self, beta, n_actions, name,
                 chkpt_dir='tmp/ddpg'):
        super(CriticNetwork, self).__init__()

        self.input_dim1 = 25
        self.fc2_dims = 30
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_ddpg')

        self.fc_in = nn.Linear(in_features=self.input_dim1, out_features=300)
        self.bn_in = nn.LayerNorm(300)

        self.fc1 = nn.Linear(in_features=300, out_features=1224)
        self.bn1 = nn.LayerNorm(1224)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding =1) # out size 40x40x12

        self.cbam = CBAM(8,reduction_ratio = 4, kernel_size = 3)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=5,stride=1, padding =1) #14x14x12
        self.flatten= nn.Flatten()
        self.bn2_flatten= nn.LayerNorm(5776)

        self.fc2 = nn.Linear(in_features=5776 + 1224, out_features=3000)
        self.bn2 = nn.LayerNorm(3000)
        # self.dropout = nn.Dropout(﻿p=0.2)
        self.fc3 = nn.Linear(in_features=3000, out_features=500)
        self.bn3 = nn.LayerNorm(500)
        self.fc4 = nn.Linear(in_features=500, out_features=30)
        self.bn4 = nn.LayerNorm(30)

        self.action_value = nn.Linear(self.n_actions, 30) #

        self.q = nn.Linear(self.fc2_dims, 1)

        # wight initialization
        f_in = 1./np.sqrt(self.fc_in.weight.data.size()[0])
        self.fc_in.weight.data.uniform_(-f_in, f_in)
        self.fc_in.bias.data.uniform_(-f_in, f_in)


        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1, f1)

        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)

        f3 = 1./np.sqrt(self.fc3.weight.data.size()[0])
        self.fc3.weight.data.uniform_(-f3, f3)
        self.fc3.bias.data.uniform_(-f3, f3)

        f4 = 1./np.sqrt(self.fc4.weight.data.size()[0])
        self.fc4.weight.data.uniform_(-f4, f4)
        self.fc4.bias.data.uniform_(-f4, f4)

        fq = 0.003
        self.q.weight.data.uniform_(-fq, fq)
        self.q.bias.data.uniform_(-fq, fq)

        fa = 1./np.sqrt(self.action_value.weight.data.size()[0])
        self.action_value.weight.data.uniform_(-fa, fa)
        self.action_value.bias.data.uniform_(-fa, fa)

        self.optimizer = optim.Adam(self.parameters(), lr=beta,
                                    weight_decay=0.01)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cuda:1')

        self.to(self.device)

    def forward(self, state1,state2, action):

        # (1) input layer
        t1 = state1 # elevation map input
        t2 = state2 #robot pose inputs

        t2 = self.fc_in(t2)
        t2 = self.bn_in(t2)
        t2 = F.relu(t2)

        t2 = self.fc1(t2)
        t2 = self.bn1(t2)
        t2= F.dropout(t2, p=0.25, training=True, inplace=False)
        t2 = F.relu(t2)

        t = self.conv1(t1)
        t = F.relu(t)

        t = self.cbam(t)

        t=self.conv2(t)
        t = F.relu(t)

        t=self.flatten(t)
        t = self.bn2_flatten(t)
        t= F.dropout(t, p=0.25, training=True, inplace=False)

        t = F.relu(t)

        combined = T.cat((t.view(t.size(0), -1),
                          t2.view(t2.size(0), -1)), dim=1)

        t = self.fc2(combined)
        t = self.bn2(t)
        t= F.dropout(t, p=0.25, training=True, inplace=False)
        t= F.relu(t)


        t = self.fc3(t)
        t = self.bn3(t)
        t= F.dropout(t, p=0.25, training=True, inplace=False)
        t = F.relu(t)

        t = self.fc4(t)
        t = self.bn4(t)

        state_value = t

        action_value = self.action_value(action)
        state_action_value = F.relu(T.add(state_value, action_value))
        #state_action_value = T.add(state_value, action_value)
        state_action_value = self.q(state_action_value)

        return state_action_value

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))

    def save_best(self):
        print('... saving best checkpoint ...')
        checkpoint_file = os.path.join(self.checkpoint_dir, self.name+'_best')
        T.save(self.state_dict(), checkpoint_file)

class ActorNetwork(nn.Module):
    def __init__(self, alpha, n_actions, name,
                 chkpt_dir='tmp/ddpg'):
        super(ActorNetwork, self).__init__()

        self.input_dim1 = 25
        self.fc2_dims = 60
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_ddpg')

        self.fc_in = nn.Linear(in_features=self.input_dim1, out_features=300)
        self.bn_in = nn.LayerNorm(300)


        self.fc1 = nn.Linear(in_features=300, out_features=1224)
        self.bn1 = nn.LayerNorm(1224)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding =1) # out size 40x40x12


        self.cbam = CBAM(8,reduction_ratio = 4, kernel_size = 3)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=5,stride=1, padding =1)

        self.flatten= nn.Flatten()
        self.bn2_flatten= nn.LayerNorm(5776)

        self.fc2 = nn.Linear(in_features=5776 + 1224, out_features=3000)

        self.bn2 = nn.LayerNorm(3000)


        self.fc3 = nn.Linear(in_features=3000, out_features=500)
        self.bn3 = nn.LayerNorm(500)
        self.fc4 = nn.Linear(in_features=500, out_features=30)
        self.bn4 = nn.LayerNorm(30)

        self.mu = nn.Linear(30, self.n_actions)
        # wight initialization
        f_in = 1./np.sqrt(self.fc_in.weight.data.size()[0])
        self.fc_in.weight.data.uniform_(-f_in, f_in)
        self.fc_in.bias.data.uniform_(-f_in, f_in)

        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)

        f3 = 1./np.sqrt(self.fc3.weight.data.size()[0])
        self.fc3.weight.data.uniform_(-f3, f3)
        self.fc3.bias.data.uniform_(-f3, f3)

        f4 = 1./np.sqrt(self.fc4.weight.data.size()[0])
        self.fc4.weight.data.uniform_(-f4, f4)
        self.fc4.bias.data.uniform_(-f4, f4)

        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1, f1)

        fmu = 0.003
        self.mu.weight.data.uniform_(-fmu, fmu)
        self.mu.bias.data.uniform_(-fmu, fmu)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cuda:1')

        self.to(self.device)

    def forward(self, state1,state2):

        # (1) input layer
        t1 = state1 # elevation map input

        t2 = state2 #robot pose inputs

        t2 = self.fc_in(t2)
        t2 = self.bn_in(t2)
        t2 = F.relu(t2)

        t2 = self.fc1(t2)
        t2 = self.bn1(t2)
        t2= F.dropout(t2, p=0.25, training=True, inplace=False)
        t2=F.relu(t2)

        t = self.conv1(t1)
        # print(t.shape)
        t_cbam_in = F.relu(t)

        t_cbam = self.cbam(t_cbam_in)

        t=self.conv2(t_cbam)
        t = F.relu(t)

        t=self.flatten(t)
        t = self.bn2_flatten(t)
        t= F.dropout(t, p=0.25, training=True, inplace=False)

        t = F.relu(t)


        combined = T.cat((t.view(t.size(0), -1),
                          t2.view(t2.size(0), -1)), dim=1)

        t = self.fc2(combined)
        t= self.bn2(t)
        t= F.dropout(t, p=0.25, training=True, inplace=False)
        t = F.relu(t)

        t = self.fc3(t)
        t= self.bn3(t)
        t= F.dropout(t, p=0.25, training=True, inplace=False)
        t = F.relu(t)

        t = self.fc4(t)
        t= self.bn4(t)
        t = F.relu(t)

        t= self.mu(t)

        t=T.tanh(t)

        return t, t_cbam, t_cbam_in

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))

    def save_best(self):
        print('... saving best checkpoint ...')
        checkpoint_file = os.path.join(self.checkpoint_dir, self.name+'_best')
        T.save(self.state_dict(), checkpoint_file)

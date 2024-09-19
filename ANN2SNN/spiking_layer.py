import torch
import torch.nn as nn

sign = True

def unsigned_spikes(model):
    for m in model.modules():
         if isinstance(m, Spiking):
             m.sign = False

#####the spiking wrapper######

class Spiking(nn.Module):
    def __init__(self, block, T, alpha_loc=2):
        super(Spiking, self).__init__()
        global sign
        self.block = block
        self.T = T
        self.idem = False
        self.sign = sign
        self.alpha_loc = alpha_loc  
    def forward(self, x):
        if self.idem:
            return x
        
        ###initialize membrane to half threshold
        threshold = self.block[self.alpha_loc].act_alpha.data
        # threshold = 1.0
        membrane = 0.5 * threshold
        sum_spikes = 0
        
        train_shape = [x.shape[0], x.shape[1]]
        x = x.flatten(0, 1)
        x = self.block[:self.alpha_loc+1](x)
        train_shape.extend(x.shape[1:])
        x = x.reshape(train_shape)
        
        #integrate charges
        for dt in range(self.T):
            membrane = membrane + x[:,dt]
            if dt == 0:
                spike_train = torch.zeros(membrane.shape[:1] + torch.Size([self.T]) + membrane.shape[1:],device=membrane.device)
                
            spikes = membrane >= threshold
            membrane[spikes] = membrane[spikes] - threshold
            spikes = spikes.float()
            sum_spikes = sum_spikes + spikes
            
            ###signed spikes###
            if self.sign:
                inhibit = membrane <= -1e-3
                inhibit = inhibit & (sum_spikes > 0)
                membrane[inhibit] = membrane[inhibit] + threshold
                inhibit = inhibit.float()
                sum_spikes = sum_spikes - inhibit
            else:
                inhibit = 0

            spike_train[:,dt] = spikes - inhibit
        spike_train = spike_train * threshold
        
        train_shape = [spike_train.shape[0], spike_train.shape[1]]
        spike_train = spike_train.flatten(0, 1)
        spike_train = self.block[self.alpha_loc+1:](spike_train)
        train_shape.extend(spike_train.shape[1:])
        spike_train = spike_train.reshape(train_shape)

        return spike_train


class Spiking_TimeEmbed(nn.Module):
    def __init__(self, block, T, alpha_loc):
        super(Spiking_TimeEmbed, self).__init__()
        global sign
        self.block = block
        self.T = T
        self.is_first = True
        self.idem = False
        self.sign = sign
        self.alpha_loc = alpha_loc
    def forward(self, x):
        if self.idem:
            return x
        
        ###initialize membrane to half threshold
        threshold = self.block[self.alpha_loc].act_alpha.data
        membrane = 0.5 * threshold
        sum_spikes = 0
        
        #prepare charges
        if self.is_first:
            x = x.unsqueeze(1)
            x = x.repeat(1, self.T)
        train_shape = [x.shape[0], x.shape[1]]
        x = x.flatten(0, 1)
        x = self.block[:self.alpha_loc+1](x)
        train_shape.extend(x.shape[1:])
        x = x.reshape(train_shape)
        
        #integrate charges
        for dt in range(self.T):
            membrane = membrane + x[:,dt]
            if dt == 0:
                spike_train = torch.zeros(membrane.shape[:1] + torch.Size([self.T]) + membrane.shape[1:],device=membrane.device)
                
            spikes = membrane >= threshold
            membrane[spikes] = membrane[spikes] - threshold
            spikes = spikes.float()
            sum_spikes = sum_spikes + spikes
            
            ###signed spikes###
            if self.sign:
                inhibit = membrane <= -1e-3
                inhibit = inhibit & (sum_spikes > 0)
                membrane[inhibit] = membrane[inhibit] + threshold
                inhibit = inhibit.float()
                sum_spikes = sum_spikes - inhibit
            else:
                inhibit = 0

            spike_train[:,dt] = spikes - inhibit
                
        spike_train = spike_train * threshold

        train_shape = [spike_train.shape[0], spike_train.shape[1]]
        spike_train = spike_train.flatten(0, 1)
        spike_train = self.block[self.alpha_loc+1:](spike_train)
        train_shape.extend(spike_train.shape[1:])
        spike_train = spike_train.reshape(train_shape)

        return spike_train


class last_Spiking(nn.Module):
    def __init__(self, block, T, alpha_loc):
        super(last_Spiking, self).__init__()
        global sign
        self.block = block
        self.T = T
        self.idem = False
        self.alpha_loc = alpha_loc
        self.sign = sign

    def forward(self, x):
        if self.idem:
            return x
        #prepare charges
        threshold = self.block[self.alpha_loc].act_alpha.data
        membrane = 0.5 * threshold
        sum_spikes = 0
        
        train_shape = [x.shape[0], x.shape[1]]
        x = x.flatten(0, 1)
        x = self.block[:self.alpha_loc+1](x)
        train_shape.extend(x.shape[1:])
        x = x.reshape(train_shape)
        
        #integrate charges
        for dt in range(self.T):
            membrane = membrane + x[:,dt]
            if dt == 0:
                spike_train = torch.zeros(membrane.shape[:1] + torch.Size([self.T]) + membrane.shape[1:],device=membrane.device)
                
            spikes = membrane >= threshold
            membrane[spikes] = membrane[spikes] - threshold
            spikes = spikes.float()
            sum_spikes = sum_spikes + spikes
            
            ###signed spikes###
            if self.sign:
                inhibit = membrane <= -1e-3
                inhibit = inhibit & (sum_spikes > 0)
                membrane[inhibit] = membrane[inhibit] + threshold
                inhibit = inhibit.float()
                sum_spikes = sum_spikes - inhibit
            else:
                inhibit = 0

            spike_train[:,dt] = spikes - inhibit
                
        spike_train = spike_train * threshold

        train_shape = [spike_train.shape[0], spike_train.shape[1]]
        spike_train = spike_train.flatten(0, 1)
        spike_train = self.block[self.alpha_loc+1:](spike_train)
        train_shape.extend(spike_train.shape[1:])
        spike_train = spike_train.reshape(train_shape)
        
        #integrate charges
        return spike_train.sum(dim=1).div(self.T)

class IF(nn.Module):
    def __init__(self):
        super(IF, self).__init__()
        ###changes threshold to act_alpha
        ###being fleet
        self.act_alpha = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x

    def show_params(self):
        act_alpha = round(self.act_alpha.data.item(), 3)
        print('clipping threshold activation alpha: {:2f}'.format(act_alpha)) 
    
    def extra_repr(self) -> str:
        return 'threshold={:.3f}'.format(self.act_alpha)  
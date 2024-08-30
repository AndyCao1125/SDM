import os
import warnings
import torch
from torchvision.utils import make_grid, save_image
import random
import numpy as np
import random
import argparse
from diffusion import GaussianDiffusionSampler
from TSM import Spk_UNet_TSM
parser = argparse.ArgumentParser()
# Gaussian Diffusion
parser.add_argument('--snn_timestep', type=int, default=8, help='snn time steps')
parser.add_argument('--img_size', type=int, default=32, help='image size')
parser.add_argument('--beta_1', default=1e-4, help='start beta value')
parser.add_argument('--beta_T', default=0.02, help='end beta value')
parser.add_argument('--T', default=1000, help='total diffusion steps')
parser.add_argument('--mean_type', default='epsilon', help='predict variable:[xprev, xstart, epsilon]')
parser.add_argument('--var_type', default='fixedlarge', help='variance type:[fixedlarge, fixedsmall]')
parser.add_argument('--ddim_step', type=int, default=50, help='sampling steps')


args = parser.parse_args()


device = torch.device('cuda:0')



def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True


def SNNsample(seed):
    # model setup

    model = Spk_UNet_TSM(
        T=1000, ch=128, ch_mult=[1, 2, 2, 4], attn=[8],
        num_res_blocks=2, dropout=0.1, timestep=args.snn_timestep, global_thres=1., use_cupy=False).cuda()


    sampler = GaussianDiffusionSampler(
        model, args.beta_1, args.beta_T, args.T, args.img_size,
        args.mean_type, args.var_type, sample_type='ddim',sample_steps=args.ddim_step).to(device)

    # load model and evaluate
    ema = False
    ckpt = torch.load(os.path.join('SDM_TSM_8T.pt'))
    
    if ema:
        print(f'Loading EMA model')
        model.load_state_dict(ckpt['ema_model'])
    else:
        print(f'Loading Origin model')
        model.load_state_dict(ckpt['net_model'], strict=True)
    

    model.eval()


    with torch.no_grad():
        path_dir = os.path.join('./')
        x_T = torch.randn(64, 3, 32, 32).to(device)

        x_0_tsm = sampler(x_T)
        
        grid = (make_grid(x_0_tsm) + 1) / 2
        save_image(grid, os.path.join(path_dir, f'ddim_tsm_8step_{seed}.png'))  


    print(f'------------Successfully Sampling!------------')


def main():

    seed=42
    seed_everything(seed)
    SNNsample(seed)

if __name__ == '__main__':
    main()

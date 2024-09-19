import os
import warnings
from absl import app, flags
from copy import deepcopy

import torch
from torchvision.datasets import CIFAR10
from torchvision.utils import make_grid, save_image
from torchvision import transforms
from tqdm import trange

from diffusion import GaussianDiffusionSampler
from model import UNet
from spiking_model import S_UNet
from score.both import get_inception_and_fid_score
import torch.nn as nn
import torch.backends.cudnn as cudnn

FLAGS = flags.FLAGS
flags.DEFINE_bool('fine_tune', False, help='fine-tune block by block')
# UNet
flags.DEFINE_integer('ch', 128, help='base channel of UNet')
flags.DEFINE_multi_integer('ch_mult', [1, 2, 2, 2], help='channel multiplier')
flags.DEFINE_multi_integer('attn', [1], help='add attention to these levels')
flags.DEFINE_integer('num_res_blocks', 2, help='# resblock in each level')
flags.DEFINE_float('dropout', 0.1, help='dropout rate of resblock')
flags.DEFINE_float('bit', 8, help='quantization parameter')
# Gaussian Diffusion
flags.DEFINE_float('beta_1', 1e-4, help='start beta value')
flags.DEFINE_float('beta_T', 0.02, help='end beta value')
flags.DEFINE_integer('T', 1000, help='total diffusion steps')
flags.DEFINE_enum('mean_type', 'epsilon', ['xprev', 'xstart', 'epsilon'], help='predict variable')
flags.DEFINE_enum('var_type', 'fixedlarge', ['fixedlarge', 'fixedsmall'], help='variance type')
# fine-tune
flags.DEFINE_float('lr', 2e-4, help='target learning rate')
flags.DEFINE_float('grad_clip', 1., help="gradient norm clipping")
flags.DEFINE_integer('ft_steps', 10000, help='total training steps')#800000
flags.DEFINE_integer('img_size', 32, help='image size')
flags.DEFINE_integer('batch_size', 128, help='batch size')
flags.DEFINE_integer('num_workers', 4, help='workers of Dataloader')
flags.DEFINE_float('ema_decay', 0.9999, help="ema decay rate")
flags.DEFINE_bool('parallel', False, help='multi gpu training')
# Logging & Sampling
flags.DEFINE_string('logdir', './logs/DDPM_CIFAR10_EPS', help='log directory')
flags.DEFINE_integer('sample_size', 64, "sampling size of images")
flags.DEFINE_integer('sample_step', 1000, help='frequency of sampling')
# Evaluation
flags.DEFINE_integer('save_step', 2000, help='frequency of saving checkpoints, 0 to disable during training')
flags.DEFINE_integer('eval_step', 0, help='frequency of evaluating model, 0 to disable during training')
flags.DEFINE_integer('num_images', 50000, help='the number of generated images for evaluation')
flags.DEFINE_bool('fid_use_torch', False, help='calculate IS and FID on gpu')
flags.DEFINE_string('fid_cache', './stats/cifar10.train.npz', help='FID cache')
flags.DEFINE_integer('gen_images', 64, help='the number of generated images for evaluation of converted-SNN')
#Fine-tune
flags.DEFINE_integer('ft_batch_size', 64, "random noise images batch-size for SNN fine tuning")


device = torch.device('cuda:0')

def adjust_learning_rate(optimizer, epoch):
    """For resnet, the lr starts from 0.1, and is divided by 10 at 80 and 120 epochs"""
    adjust_list = [150, 225]
    if epoch in adjust_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1

def model_update(agent, target):#for every iter,update spiking models according to agent models
    old_state_dict = agent.state_dict()
    new_state_dict = {}

    for key in old_state_dict:
        parts = key.split('.')
        if 'main' in key or 'shortcut' in key or 'head' in key:
            new_key = key
            
        else:
            parts.insert(-2, 'block')
            new_key = '.'.join(parts)
    
        new_state_dict[new_key] = old_state_dict[key]
        
    target.load_state_dict(new_state_dict, strict=False)

def model_load_acta(model, spiking = False):
    new_state_dict = {}

    old_state_dict = torch.load(FLAGS.logdir+'/ckpt.pt')['ema_model']
    
    if spiking:
       
        for key in old_state_dict:
            parts = key.split('.')
            if 'main' in key or 'shortcut' in key or 'head' in key:
                new_key = key
                
            else:
                parts.insert(-2, 'block')
                new_key = '.'.join(parts)

            new_state_dict[new_key] = old_state_dict[key]
            
        model.load_state_dict(new_state_dict, strict=False)
    else:    
        model.load_state_dict(old_state_dict, strict=True)
    

def para_ls_make():
    state_dict = torch.load(FLAGS.logdir+'/ckpt.pt')['ema_model']
    block_nums = {
        'downblocks':0,
        'middleblocks':0,
        'upblocks':0,
    }
    para_ls = {
        'time_embedding':['time_embedding.timembedding'],
        'head':['head'],
        'downblocks':[],
        'middleblocks':[],
        'upblocks':[],
        'tail':['tail'],
    }

    for k in state_dict.keys():
        k2 = k.split('.')
        if k2[0] == 'downblocks' or k2[0] == 'middleblocks' or k2[0] == 'upblocks':
            if '.'.join(k2[:2]) not in para_ls[k2[0]]:
                para_ls[k2[0]].append('.'.join(k2[:2]))

    block_nums['downblocks'] = len(para_ls['downblocks'])
    block_nums['middleblocks'] = len(para_ls['middleblocks'])
    block_nums['upblocks'] = len(para_ls['upblocks'])

    return para_ls, block_nums
    

def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay +
            source_dict[key].data * (1 - decay))


def infiniteloop(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x


def warmup_lr(step):
    return min(step, FLAGS.warmup) / FLAGS.warmup


def evaluate(sampler, model):
    model.eval()
    with torch.no_grad():
        images = []
        desc = "generating images"
        for i in trange(0, FLAGS.num_images, FLAGS.batch_size, desc=desc):
            batch_size = min(FLAGS.batch_size, FLAGS.num_images - i)
            x_T = torch.randn((batch_size, 3, FLAGS.img_size, FLAGS.img_size))
            batch_images = sampler(x_T.to(device)).cpu()
            images.append((batch_images + 1) / 2)
        images = torch.cat(images, dim=0).numpy()
    model.train()
    (IS, IS_std), FID = get_inception_and_fid_score(
        images, FLAGS.fid_cache, num_images=FLAGS.num_images,
        use_torch=FLAGS.fid_use_torch, verbose=True)
    return (IS, IS_std), FID, images

def infiniteloop(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x

def fine_tune():
    print('=> Building model...')
    model=None
    model = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout, bit=FLAGS.bit)
    
    net = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout, bit=FLAGS.bit)
    
    snn = S_UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout, bit=FLAGS.bit, spike_time = 2**int(FLAGS.bit)-1)
    
    snn_eval = deepcopy(snn)
    model = model.to(device)
    net = net.to(device)
    snn = snn.to(device)
    snn_eval = snn_eval.to(device)
    
    if FLAGS.parallel:
        model = torch.nn.DataParallel(model)
        net = torch.nn.DataParallel(net)
        snn = torch.nn.DataParallel(snn)
        snn_eval = torch.nn.DataParallel(snn_eval)
    
    cudnn.benchmark = True
        
    x_T = torch.randn(FLAGS.sample_size, 3, FLAGS.img_size, FLAGS.img_size)
    x_T = x_T.to(device)
    
    if not os.path.exists(str(FLAGS.logdir)):
        os.makedirs(str(FLAGS.logdir))
    fdir = str(FLAGS.logdir)+'/'+'SNN'+'_'+str(FLAGS.bit)+'bit_ft'
    if not os.path.exists(fdir):
        os.makedirs(fdir)
        
    if os.path.isfile(FLAGS.logdir + '/ckpt.pt'):
        model_load_acta(model)
        model_load_acta(snn, spiking=True)
        model_load_acta(net)
    else:
        print('No pre-trained model found !')
        exit()

    # dataset
    dataset = CIFAR10(
        root='./data', train=True, download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=FLAGS.batch_size, shuffle=True,
        num_workers=FLAGS.num_workers, drop_last=True)
    datalooper = infiniteloop(dataloader)
    print('=> loading data...')
    
    if not os.path.exists(os.path.join(fdir, 'ft_blocks_samples')):
        os.makedirs(os.path.join(fdir, 'ft_blocks_samples'))
    model_save_path = os.path.join(fdir, 'model_backpacks')
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    ##prepraration starts
    para_ls, block_nums = para_ls_make()#load U-NET structures as a dict
    

    duration =  2**FLAGS.bit - 1    
    # segments = para_ls.keys()
    # af_names = ['block1.1.act_alpha','temb_proj.0.act_alpha','block2.1.act_alpha']
    # tail_af_names = '1.act_alpha'
    downsample_ls = ['downblocks.2', 'downblocks.5', 'downblocks.8']
    upsample_ls = ['upblocks.3', 'upblocks.7', 'upblocks.11']


    model.eval()
    snn.eval()
    
    bypass_blocks(model, para_ls)
    model.tail_idem = True
    
    bypass_blocks(snn, para_ls)
    snn.tail.idem = True
    criterion = nn.MSELoss()
    
    ###preparation ends
    
    ##
    for segment in [q for q in para_ls.keys() if q != 'tail']:
        for (block_idx, block) in enumerate(para_ls[segment]):
            print('=======We are tuning Segment: %s Block: %s ==========' %(segment, block))
            
            #set reference and tuner
            if segment == 'time_embedding':
                m = getattr(getattr(model, segment), 'timembedding')
                model.timembedding_idem = False
                
                m = getattr(net, segment)
                m = getattr(m, 'timembedding')
                tuner = m
                net.timembedding_idem = False
                getattr(getattr(tuner,'2'),'act_alpha').requires_grad_(False)#ignore para act_alpha's grads
            elif segment == 'head':
                m = getattr(model,'head')
                model.head_idem = False

                m = getattr(net,'head')
                tuner = m
                net.head_idem = False
            else:
                m = getattr(getattr(model, segment),str(block_idx))
                m.idem = False

                m = getattr(getattr(net, segment),str(block_idx))
                tuner = m
                tuner.idem = False

                if block not in downsample_ls + upsample_ls:
                    #downblocks.0.block1.1.act_alpha
                    #downblocks.0.temb_proj.0.act_alpha
                    #downblocks.0.block2.1.act_alpha
                    getattr(getattr(getattr(tuner,'block1'),'1'),'act_alpha').requires_grad_(False)
                    getattr(getattr(getattr(tuner,'temb_proj'),'0'),'act_alpha').requires_grad_(False)
                    getattr(getattr(getattr(tuner,'block2'),'1'),'act_alpha').requires_grad_(False)

            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, tuner.parameters()), lr=FLAGS.lr)


            #backup states of current block
            if segment == 'time_embedding':
                m = getattr(getattr(snn, segment), 'timembedding')
            elif segment == 'head':
                m = getattr(snn,'head')
            else:
                m = getattr(getattr(snn, segment),str(block_idx))    
            record = m.state_dict()        
            for k, v in record.items():
                record[k] = v.cpu()        
            
            ft_steps = FLAGS.ft_steps
            if segment in ['time_embedding', 'head']:#don't need to fine-tune these two blocks for there are no loss
                ft_steps = 1
            
            with trange(ft_steps, dynamic_ncols=True) as pbar:
                for step in pbar:
                    #fine-tune
                    input = next(datalooper).to(device)
                    t = torch.randint(int(duration), size=(input.shape[0], ), device=input.device)

                    input = input.to(device)
                    with torch.no_grad():
                        model.upblocks_target = True
                        target_map = model(input, t)
                        model.upblocks_target = False
                    if segment == 'time_embedding':
                        m = getattr(getattr(snn, segment), 'timembedding')
                        snn.timembedding_idem = True
                    elif segment == 'head':
                        m = getattr(snn,'head')
                        snn.head_idem = True
                    else:
                        m = getattr(getattr(snn, segment),str(block_idx))
                        m.idem = True

                    in_maps = snn(input, t)
                    
                    if segment == 'time_embedding':
                        snn.timembedding_idem = False
                    elif segment == 'head':
                        snn.head_idem = False
                    else:
                        m.idem = False

                    if segment not in ['time_embedding', 'head']:
                        k = getattr(getattr(snn, 'time_embedding'), 'timembedding')
                        temb = k(t)

                    if segment == 'time_embedding':
                        out_maps = m(in_maps)
                    elif segment == 'head':
                        in_maps = input
                        in_maps2 = in_maps.detach()
                        in_maps2 = in_maps2.unsqueeze(1)
                        in_maps2 = in_maps2.repeat(1, int(duration), 1, 1, 1)
                        train_shape = [in_maps2.shape[0], in_maps2.shape[1]]
                        in_maps2 = in_maps2.flatten(0, 1)
                        out_maps = m(in_maps2)
                        train_shape.extend(out_maps.shape[1:])
                        out_maps = out_maps.reshape(train_shape)
                    else:
                        out_maps = m(in_maps, temb)

                    if segment != 'time_embedding' and segment != 'head':
                        in_maps = in_maps.sum(1).div(duration)
                    out_maps = out_maps.sum(1).div(duration)
                    if segment not in ['time_embedding', 'head']:
                        temb = temb.sum(1).div(duration)

                    if segment in ['time_embedding', 'head']:
                        output = tuner(in_maps)
                    else:
                        output = tuner(in_maps, temb)
                    
                    output.data = out_maps.data
                    loss = criterion(output, target_map)         
                    pbar.set_postfix(loss='%.6f' % loss)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    
                    #update weights
                    model_update(tuner, m)
                        
                    if segment == 'time_embedding':
                        m = getattr(getattr(snn, segment), 'timembedding')
                        snn.timembedding_idem = False
                    elif segment == 'head':
                        m = getattr(snn,'head')
                        snn.head_idem = False
                    else:
                        m = getattr(getattr(snn, segment),str(block_idx))
                        m.idem = False
                    
                    if segment == 'time_embedding':
                         m = getattr(getattr(snn, segment), 'timembedding')
                    elif segment == 'head':
                        m = getattr(snn,'head')
                    else:
                        m = getattr(getattr(snn, segment),str(block_idx))       
                    record = m.state_dict()        
                    for k, v in record.items():
                        record[k] = v.cpu()  
                    
                    if (step + 1) % FLAGS.save_step == 0:
                        snn_eval.load_state_dict(snn.state_dict())
                        switch_on(snn_eval, para_ls)
                        #save fine-tuned snn model
                        torch.save({
                            'ema_model': snn_eval.state_dict(),
                        }, os.path.join(model_save_path, '{}_{}_model.pt'.format(block,str(step))))

                        #sample and validate
                        sampler = GaussianDiffusionSampler(
                            snn_eval, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, img_size=FLAGS.img_size,
                            mean_type=FLAGS.mean_type, var_type=FLAGS.var_type).to(device)
                        if FLAGS.parallel:
                            sampler = torch.nn.DataParallel(sampler)
                        snn_eval.eval()
                        with torch.no_grad():
                            x_0 = sampler(x_T)
                            grid = (make_grid(x_0) + 1) / 2
                            path = os.path.join(
                                            fdir, 'ft_blocks_samples', '{}_{}_result.png'.format(block,str(step)))
                            save_image(grid, path)


                print('Update...')        
                #revert
                if segment == 'time_embedding':
                    m = getattr(getattr(snn, segment), 'timembedding')
                elif segment == 'head':
                    m = getattr(snn,'head')
                else:
                    m = getattr(getattr(snn, segment),str(block_idx))
                m.load_state_dict(record)

    #enable tail block
    snn.tail.idem = False


         
def switch_on(model, para_ls):
    for segment in [s for s in para_ls.keys() if s != 'tail']:
        if segment == 'time_embedding':
            model.timembedding_idem = False
        elif segment == 'head':
            model.head_idem = False
        else:    
            for (idx,_) in enumerate(para_ls[segment]):
                getattr(getattr(model,segment),str(idx)).idem = False
    model.tail_idem = False
       
def switch_off(model, para_ls):
    for segment in [s for s in para_ls.keys() if s != 'tail']:
        if segment == 'time_embedding':
            model.timembedding_idem = True
        elif segment == 'head':
            model.head_idem = True
        else:    
            for (idx,_) in enumerate(para_ls[segment]):
                getattr(getattr(model,segment),str(idx)).idem = True
    model.tail_idem = True

def bypass_blocks(model, para_ls):
    for segment in [s for s in para_ls.keys() if s != 'tail']:
        if segment == 'time_embedding':
            model.timembedding_idem = True
        elif segment == 'head':
            model.head_idem = True
        else:    
            for (idx,_) in enumerate(para_ls[segment]):
                getattr(getattr(model,segment),str(idx)).idem = True


def main(argv):
    # suppress annoying inception_v3 initialization warning
    warnings.simplefilter(action='ignore', category=FutureWarning)
    if FLAGS.fine_tune:
        fine_tune()
    if not FLAGS.fine_tune:
        print('Add --fine_tune to execute the corresponding task')


if __name__ == '__main__':
    app.run(main)
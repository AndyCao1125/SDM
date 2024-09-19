import os
import warnings
from absl import app, flags
import torch
from torchvision.utils import make_grid, save_image
from tqdm import trange

from diffusion import GaussianDiffusionSampler
from spiking_model import S_UNet
from score.both import get_inception_and_fid_score

FLAGS = flags.FLAGS
flags.DEFINE_bool('eval', False, help='load ckpt.pt and evaluate FID and IS')
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

flags.DEFINE_integer('img_size', 32, help='image size')
flags.DEFINE_integer('batch_size', 128, help='batch size')
flags.DEFINE_integer('num_workers', 4, help='workers of Dataloader')
flags.DEFINE_bool('parallel', False, help='multi gpu training')
# Logging
flags.DEFINE_string('logdir', './logs/DDPM_CIFAR10_EPS', help='log directory')

flags.DEFINE_integer('num_images', 50000, help='the number of generated images for evaluation')
flags.DEFINE_bool('fid_use_torch', False, help='calculate IS and FID on gpu')
flags.DEFINE_string('fid_cache', './stats/cifar10.train.npz', help='FID cache')
flags.DEFINE_integer('gen_images', 64, help='the number of generated images for evaluation of converted-SNN')

device = torch.device('cuda:0')

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

def eval():
    print("Start converting...")
    
    if not os.path.exists(os.path.join(FLAGS.logdir, 'images')):
        os.makedirs(os.path.join(FLAGS.logdir, 'images'))

    model = S_UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout, bit=FLAGS.bit, spike_time = 2**int(FLAGS.bit)-1)

    new_state_dict = {}
    old_state_dict = torch.load(FLAGS.logdir+'/ckpt.pt')['ema_model']
    
    for key in old_state_dict:
        parts = key.split('.')
        if 'main' in key or 'shortcut' in key or 'head' in key:
            new_key = key
        else:
            parts.insert(-2, 'block')
            new_key = '.'.join(parts)
        
        new_state_dict[new_key] = old_state_dict[key]
        
    model.load_state_dict(new_state_dict, strict=False)
    
    print("Successfully load weight!")
    sampler = GaussianDiffusionSampler(
        model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, img_size=FLAGS.img_size,
        mean_type=FLAGS.mean_type, var_type=FLAGS.var_type).to(device)
    if FLAGS.parallel:
        sampler = torch.nn.DataParallel(sampler)
    x_T = torch.randn(FLAGS.gen_images, 3, FLAGS.img_size, FLAGS.img_size).to(device)
    with torch.no_grad():
        x_0 = sampler(x_T)
        grid = (make_grid(x_0) + 1) / 2
        save_image(grid, FLAGS.logdir + '/images'+'/SDDPM_T'+str(2**int(FLAGS.bit)-1)+'.png')
    
    torch.save({'ema_model': model.state_dict()}, os.path.join(FLAGS.logdir, 'converted_snn.pt'))
    print('Convert successfully.')

def main(argv):
    # suppress annoying inception_v3 initialization warning
    warnings.simplefilter(action='ignore', category=FutureWarning)
    if FLAGS.eval:
        eval()
    if not FLAGS.eval:
        print('Add --eval to execute corresponding tasks')


if __name__ == '__main__':
    app.run(main)

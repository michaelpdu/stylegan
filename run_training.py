# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

import argparse
import copy
import os
import sys

import dnnlib
from dnnlib import EasyDict

import config
from metrics import metric_base

#----------------------------------------------------------------------------

def run(dataset, data_dir, result_dir, config_id, num_gpus, total_kimg, gamma, mirror_augment, pkl_path):
    train     = EasyDict(run_func_name='training.training_loop.training_loop')              # Options for training loop.
    G         = EasyDict(func_name='training.networks_stylegan.G_style')                    # Options for generator network.
    D         = EasyDict(func_name='training.networks_stylegan.D_basic')                    # Options for discriminator network.
    G_opt     = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8)                               # Options for generator optimizer.
    D_opt     = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8)                               # Options for discriminator optimizer.
    G_loss    = EasyDict(func_name='training.loss.G_logistic_nonsaturating')                # Options for generator loss.
    D_loss    = EasyDict(func_name='training.loss.D_logistic_simplegp', r1_gamma=10.0)      # Options for discriminator loss.
    sched     = EasyDict()                                                                  # Options for TrainingSchedule.
    grid      = EasyDict(size='4k', layout='random')                                        # Options for setup_snapshot_image_grid().
    metrics   = [metric_base.fid50k]                                                        # Options for MetricGroup.
    sc        = dnnlib.SubmitConfig()                                                       # Options for dnnlib.submit_run().
    tf_config = {'rnd.np_random_seed': 1000}                                                # Options for tflib.init_tf().

    train.data_dir = data_dir
    train.total_kimg = total_kimg
    train.mirror_augment = mirror_augment
    train.image_snapshot_ticks = train.network_snapshot_ticks = 10
    if pkl_path == '':
        print('pickle path is empty string.')
        train.resume_run_id = None
    else:
        print('pickle path:', pkl_path)
        train.resume_run_id = pkl_path
    
    desc = 'sgan'
    desc += '-' + dataset
    dataset_args = EasyDict(tfrecord_dir=dataset)

    assert num_gpus in [1, 2, 4, 8]
    sc.num_gpus = num_gpus
    desc += '-%dgpu' % num_gpus

    sched.lod_initial_resolution = 8
    sched.G_lrate_dict = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}
    sched.D_lrate_dict = EasyDict(sched.G_lrate_dict)
    if num_gpus == 8:
        sched.minibatch_base = 32
        sched.minibatch_dict = {4: 512, 8: 256, 16: 128, 32: 64, 64: 32}
    elif num_gpus == 4:
        sched.minibatch_base = 16
        sched.minibatch_dict = {4: 512, 8: 256, 16: 128, 32: 64, 64: 32, 128: 16}

    kwargs = EasyDict(train)
    kwargs.update(G_args=G, D_args=D, G_opt_args=G_opt, D_opt_args=D_opt, G_loss_args=G_loss, D_loss_args=D_loss)
    kwargs.update(dataset_args=dataset_args, sched_args=sched, grid_args=grid, metric_arg_list=metrics, tf_config=tf_config)
    kwargs.submit_config = copy.deepcopy(sc)
    kwargs.submit_config.run_dir_root = dnnlib.submission.submit.get_template_from_path(config.result_dir)
    kwargs.submit_config.run_dir_ignore += config.run_dir_ignore
    kwargs.submit_config.run_desc = desc
    dnnlib.submit_run(**kwargs)
#----------------------------------------------------------------------------

def _str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def _parse_comma_sep(s):
    if s is None or s.lower() == 'none' or s == '':
        return []
    return s.split(',')

#----------------------------------------------------------------------------

_examples = '''examples:

  # Train StyleGAN2 using the FFHQ dataset
  python %(prog)s --num-gpus=4 --data-dir=~/datasets --dataset=yellow-real --mirror-augment=true
'''

def main():
    parser = argparse.ArgumentParser(
        description='Train StyleGAN2.',
        epilog=_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')
    parser.add_argument('--data-dir', help='Dataset root directory', required=True)
    parser.add_argument('--dataset', help='Training dataset', required=True)
    parser.add_argument('--num-gpus', help='Number of GPUs (default: %(default)s)', default=1, type=int, metavar='N')
    parser.add_argument('--total-kimg', help='Training length in thousands of images (default: %(default)s)', metavar='KIMG', default=25000, type=int)
    parser.add_argument('--mirror-augment', help='Mirror augment (default: %(default)s)', default=True, metavar='BOOL', type=_str_to_bool)
    parser.add_argument('--pkl-path', help='Path to network pickle file', default='')

    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        print ('Error: dataset root directory does not exist.')
        sys.exit(1)
    
    run(**vars(args))

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------


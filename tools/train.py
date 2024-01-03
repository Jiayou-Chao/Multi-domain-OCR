# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.registry import RUNNERS
from mmengine.runner import Runner


import re
import torch

def count_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total_params = trainable_params + non_trainable_params
    print(f'Total parameters: {total_params:,d}')
    print(f'Trainable parameters: {trainable_params:,d}')
    print(f'Non-trainable parameters: {non_trainable_params:,d}')
    return {'total_params': total_params, 'trainable_params': trainable_params, 'non_trainable_params': non_trainable_params}

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('config', help='Train config file path')
    parser.add_argument('--work-dir', help='The dir to save logs and models')
    parser.add_argument(
        '--resume', action='store_true', help='Whether to resume checkpoint.')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='Enable automatic-mixed-precision training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='Whether to scale the learning rate automatically. It requires '
        '`auto_scale_lr` in config, and `base_batch_size` in `auto_scale_lr`')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='Job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

def not_is_adapter_module(name, module):
    if (isinstance(module, torch.nn.Conv2d) and (module.kernel_size == (3, 3)) and 'backbone' in name):
        return True
    if 'backbone' in name:
        return False
    if isinstance(module, (torch.nn.LayerNorm, torch.nn.BatchNorm2d)):
        return False
    if (('adapters_list' in name) and 'decoder' in name):
        return False
    if 'bottleneck' in name:
        return False
    if 'norm' in name:
        return False
    if '.bn' in name:
        return False
    if 'classifier' in name:
        return False
    if isinstance(module, torch.nn.Linear):
        return True
    return False


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    # enable automatic-mixed-precision training
    if args.amp:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    if args.resume:
        cfg.resume = True

    # enable automatically scaling LR
    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Can not find "auto_scale_lr" or '
                               '"auto_scale_lr.base_batch_size" in your'
                               ' configuration file.')

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)
        
    # if cfg.get('adapter_load_from', False):
    #     import torch
    #     paramwise_cfg = dict(custom_keys={})
    #     cpt = runner.load_checkpoint(cfg.adapter_load_from)
    #     # for name, module in runner.model.named_modules():
    #     #     if '.' in re.sub(r'^module.', '', name):
    #     #         paramwise_cfg['custom_keys'][re.sub(
    #     #             r'^module.', '', name)] = dict(lr_mult=0, decay_mult=0) 
    #     for name, module in runner.model.named_modules():
    #         if isinstance(module, torch.nn.Conv2d) and (module.in_channels
    #                                                     == 3):
    #             module.requires_grad = False
    #             module.requires_grad_(False)
    #             paramwise_cfg['custom_keys'][re.sub(
    #                 r'^module.', '', name)] = dict(lr_mult=0.0, decay_mult=0.0)
    #     # cfg.optim_wrapper.optimizer['paramwise_cfg'] = paramwise_cfg
    #     cfg.optim_wrapper['paramwise_cfg'] = paramwise_cfg
    #     runner = Runner.from_cfg(cfg)
    #     for name, module in runner.model.named_modules():
    #         if isinstance(module, torch.nn.Conv2d) and (module.in_channels
    #                                                     == 3):
    #             module.requires_grad = False
    #             module.requires_grad_(False)
    #             paramwise_cfg['custom_keys'][re.sub(
    #                 r'^module.', '', name)] = dict(lr_mult=0.0, decay_mult=0.0)
    #             print_log(f'Freeze {name} in model', logger='current')
    # assert cfg.get('train_mode', False) in ['adapter', 'full']
    # print_log(f'num of parameters: {count_parameters(runner.model)}', logger='current')
    if cfg.get('train_mode', False) == 'adapter':
        paramwise_cfg = dict(custom_keys={})
        for name, module in runner.model.named_modules():
            if not_is_adapter_module(name, module):
                # module.requires_grad = False
                module.requires_grad_(False)
                paramwise_cfg['custom_keys'][re.sub(
                    r'^module.', '', name)] = dict(lr_mult=0.0, decay_mult=0.0)
                print(f'Freeze {name} in model')

        # cfg.optim_wrapper.optimizer['paramwise_cfg'] = paramwise_cfg
        cfg.optim_wrapper['paramwise_cfg'] = paramwise_cfg
        print_log(f'num of parameters: {count_parameters(runner.model)}', logger='current')
        runner = Runner.from_cfg(cfg)
        for name, module in runner.model.named_modules():
            if not_is_adapter_module(name, module):
                module.requires_grad = False
                module.requires_grad_(False)
                paramwise_cfg['custom_keys'][re.sub(
                    r'^module.', '', name)] = dict(lr_mult=0.0, decay_mult=0.0)
                # print_log(f'Freeze {name} in model', logger='current')
                print_log(f'Freeze {name} in model: {count_parameters(module)["non_trainable_params"]}', logger='current')
            else:
                # print(f'Not freeze {name} in model')
                print_log(f'Not freeze {name} in model: {count_parameters(module)["trainable_params"]}', logger='current')
    print_log(f'num of parameters: {count_parameters(runner.model)}', logger='current')
        
# t2=set(dict(runner.model.named_modules()))
# t1 = set(cpt.get('state_dict', {}).keys())
# print(t2-t1)
# print(t1-t2)
    # start training
    runner.train()


if __name__ == '__main__':
    main()

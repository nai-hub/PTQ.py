import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.join(curPath, '../')
sys.path.append(rootPath)

import torch
import argparse
import time
import warnings
import mmcv
from mmcv import DictAction, Config
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint, wrap_fp16_model)
from mmcls.apis import init_random_seed, set_random_seed
from mmcls.models import build_classifier
from mmcls.utils import collect_env
from mmdet.models import build_detector
from src.utils import get_root_logger
from src.quant.api.quant_model import quant_model


def parse_args():
    parser = argparse.ArgumentParser(description='Pytorch quantizer')
    parser.add_argument('--config',
                        default='../classification/configs/resnet18/resnet18_8xb16_cifar10.py',
                        help='config file')
    parser.add_argument('--checkpoint',
                        default='../classification/checkpoint/resnet18_b16x8_cifar10_20210528-bd6371c8.pth',
                        help='checkpoint path by terminal running')

    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')

    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='(Deprecated, please use --gpu-id) number of gpus to use '
             '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
             '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
             '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--device', default='cpu', help='device used for testing. (Deprecated)')

    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')

    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')

    parser.add_argument(
        '--metrics',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., '
             '"accuracy", "precision", "recall", "f1_score", "support" for single '
             'label dataset, and "mAP", "CP", "CR", "CF1", "OP", "OR", "OF1" for '
             'multi-label dataset , and "bbox", "segm", "proposal" for COCO, '
             'and "mAP","recall" for PASCAL VOC')
    parser.add_argument(
        '--metric-options',
        nargs='+',
        action=DictAction,
        default={},
        help='custom options for evaluation, the key-value pair in xxx=yyy '
             'format will be parsed as a dict metric_options for dataset.evaluate()'
             ' function.')

    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    if args.config is not None:
        cfg = Config.fromfile(args.config)
    else:
        raise Exception('Need config file.')

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if args.gpus is not None:
        cfg.gpu_ids = range(1)
        warnings.warn('`--gpus` is deprecated because we only support '
                      'single GPU mode in non-distributed training. '
                      'Use `gpus=1` now.')
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed training. Use the first GPU '
                      'in `gpu_ids` now.')
    if args.gpus is None and args.gpu_ids is None:
        cfg.gpu_ids = [args.gpu_id]

    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    cfg.no_validate = args.no_validate

    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    cfg.timestamp = timestamp
    # log path
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is not None:
        cfg.work_dir = cfg.get('work_dir', None)
    else:
        cfg.work_dir = os.path.join('./work_dir',
                                    os.path.splitext(os.path.basename(args.config))[0] + '_' + f'{timestamp}')

    # create work_dir
    mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))
    # init the logger before other steps
    log_file = os.path.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    # set random seeds
    seed = init_random_seed(args.seed)
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    # create model and dataset
    if cfg.target_type == 'classification':
        logger.info('Create Classification Network.')

        model = build_classifier(cfg.model)

        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)

        if args.checkpoint is not None:
            checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

            if 'CLASSES' in checkpoint.get('meta', {}):
                CLASSES = checkpoint['meta']['CLASSES']
                model.CLASSES = CLASSES
            else:
                from mmcls.datasets import ImageNet
                warnings.simplefilter('once')
                warnings.warn('Class names are not saved in the checkpoints\'s '
                              'meta data, use imagenet by default.')
                model.CLASSES = ImageNet.CLASSES
        else:
            model.init_weights()
            raise Exception('Checkpoint is None.')

    elif cfg.target_type == 'detection':

        logger.info('Create Detection Network.')
        # build the model and load checkpoint
        model = build_detector(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg'))

        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
        if args.checkpoint is not None:
            logger.info('load checkpoint')
            checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
            # old versions did not save class info in checkpoints, this walkaround is
            # for backward compatibility
            if 'CLASSES' in checkpoint.get('meta', {}):
                model.CLASSES = checkpoint['meta']['CLASSES']
            else:
                model.CLASSES = None
        else:
            model.init_weights()
            raise Exception('Checkpoint is None.')
    else:
        raise Exception('Only support classification and detection target types.')
    # prune model
    logger.info('-' * 25 + 'Start Distiller' + '-' * 23)
    cfg.device = args.device

    quant_model(model,
                  cfg,
                  distributed=distributed,
                  device='cpu' if args.device == 'cpu' else 'cuda',
                  meta=meta
                  )

    # remove handler  
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    return


if __name__ == '__main__':
    main()

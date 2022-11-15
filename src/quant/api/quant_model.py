import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.join(curPath, '../../../')
sys.path.append(rootPath)

import warnings
import torch.nn as nn

import mmcv
from mmcv.parallel import MMDataParallel

from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets import build_dataset as build_dataset_for_det
from mmdet.datasets import build_dataloader as build_dataloader_for_det

from mmcls.datasets import build_dataset as build_dataset_for_cls
from mmcls.datasets import build_dataloader as build_dataloader_for_cls

from src.quant.core.PTQ import PostTrainingStaticQuantization
from src.utils import get_root_logger


def quant_model(model,
                cfg,
                distributed=False,
                device='cpu',
                meta=None):
    assert isinstance(model, nn.Module), f'Model is not {type(model)}.'
    assert isinstance(cfg, mmcv.utils.config.Config)
    assert distributed is False, 'Not Support Distribute'

    logger = get_root_logger(log_level=cfg.log_level)

    if cfg.target_type == 'classification':

        # test loader
        test_dataset = build_dataset_for_cls(cfg.data.test, default_args=dict(test_mode=True))
        test_dataloader = build_dataloader_for_cls(
            test_dataset,
            samples_per_gpu=cfg.data.samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False,
            round_up=True)

        # train loader
        calibration_dataset = build_dataset_for_cls(cfg.data.train)
        sampler_cfg = cfg.data.get('sampler', None)
        calibration_dataloader = build_dataloader_for_cls(
            calibration_dataset,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            num_gpus=len(cfg.gpu_ids),
            dist=distributed,
            round_up=True,
            seed=cfg.seed,
            sampler_cfg=sampler_cfg)

    elif cfg.target_type == 'detection':
        # build the dataloader
        # in case the test dataset is concatenated
        samples_per_gpu = 1
        if isinstance(cfg.data.test, dict):
            cfg.data.test.test_mode = True
            samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
            if samples_per_gpu > 1:
                # Replace 'ImageToTensor' to 'DefaultFormatBundle'
                cfg.data.test.pipeline = replace_ImageToTensor(
                    cfg.data.test.pipeline)
        elif isinstance(cfg.data.test, list):
            for ds_cfg in cfg.data.test:
                ds_cfg.test_mode = True
            samples_per_gpu = max(
                [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
            if samples_per_gpu > 1:
                for ds_cfg in cfg.data.test:
                    ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

        # test loader           
        test_dataset = build_dataset_for_det(cfg.data.test)
        test_dataloader = build_dataloader_for_det(
            test_dataset,
            samples_per_gpu=samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)

        # calibration loader
        calibration_dataset = build_dataset_for_det(cfg.data.calibration)
        model.CLASSES = calibration_dataset.CLASSES

        calibration_dataloader = build_dataloader_for_det(
            calibration_dataset,
            samples_per_gpu=samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)

    else:
        raise Exception('target_type is error.')

    # put model on gpus
    if distributed:
        raise NotImplementedError
    else:
        if device == 'cpu':
            warnings.warn(
                'The argument `device` is deprecated. To use cpu to train, '
                'please refers to https://mmclassification.readthedocs.io/en'
                '/latest/getting_started.html#train-a-model')
            model = model.cpu()
        else:
            model = MMDataParallel(model, device_ids=cfg.gpu_ids)
            if not model.device_ids:
                from mmcv import __version__, digit_version
                assert digit_version(__version__) >= (1, 4, 4), \
                    'To train with CPU, please confirm your mmcv version ' \
                    'is not lower than v1.4.4'

    PTQ = PostTrainingStaticQuantization(model,
                                         cfg,
                                         logger,
                                         test_dataloader,
                                         calibration_dataloader=calibration_dataloader,
                                         distributed=distributed,
                                         device=device)

    PTQ.quantization()

    return

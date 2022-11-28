import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.join(curPath, '../../../')
sys.path.append(rootPath)

import torch
import torch.nn as nn
import copy
import warnings
import pulp
from pulp import *
import time
import numpy as np
from numbers import Number

import mmcv
from mmcv.runner import get_dist_info
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcls.apis import single_gpu_test as single_gpu_test_cls
from mmdet.apis import single_gpu_test as single_gpu_test_det

from src.utils import *
from src.quant.core.quanter.linear_quanter import *
from src.utils.state import quant_prepare, quant_convert


class PostTrainingStaticQuantization(object):
    def __init__(self,
                 model,
                 cfg,
                 logger,
                 test_dataloader,
                 calibration_dataloader=None,
                 distributed=False,
                 device='cpu'):
        """
        Reference : [1] 'Improving PTQ: Layer-wise Calibration and Integer Programming'
                    [2] 'HAWQ'
        """

        assert isinstance(model, nn.Module)
        assert isinstance(cfg, mmcv.utils.config.Config)
        assert isinstance(test_dataloader, torch.utils.data.dataloader.DataLoader)
        assert distributed == False, 'Distribute NotImplemented'

        if logger is None:
            self.logger = get_root_logger()
        else:
            self.logger = logger

        self.model = copy.deepcopy(model)
        self.test_loader = test_dataloader
        self.calibration_dataloader = calibration_dataloader
        self.cfg = cfg
        self.distributed = distributed
        self.device = device

        if self.calibration_dataloader:
            self._calibration() # 将模型标准化为量化模型

        self.org_model_size = get_model_size(self.model)
        self.org_params_size = get_params_size(self.model) * 4.  # MB

        if not self.cfg.selfdistill_model:
            self.input_shape = []
            for i, data in enumerate(self.test_loader):
                self.input_shape.append(data['img'][0].size()[-1])
                break

            self.org_FLOPs = get_flops(self.model, self.input_shape)
            self.org_BOPs = 32 * 32 * self.org_FLOPs / 1e6  # M
        self.model = self.put_model_on_gpus(self.model)

        self.org_acc = self._test_model(self.model)

        self.logger.info('{:<20s} : {}'.format('original acc', self.org_acc))
        self.logger.info('{:<20s} : {:.2f}MB'.format('original model size', self.org_model_size))
        self.logger.info('{:<20s} : {:.2f}MB'.format('original params size', self.org_params_size))

        if not self.cfg.selfdistill_model:
            self.logger.info('{:<20s} : {:.2f}M'.format('original BOPs', self.org_BOPs))

    def _get_quantable_layer_index(self):

        self.params_layer_list = []
        self.quant_layer_idx = []
        self.layer_type_info = dict()

        for i, m in enumerate(self.model.modules()):
            layer_str = str(m)
            layer_name = layer_str[:layer_str.find('(')].strip()
            if layer_name in ['Conv2d', 'Linear']:
                for param in m.named_parameters():
                    if 'bias' in param[0]:  # ignore bias quantization
                        continue
                    if not param[1].requires_grad:  # ignore param not require graduation
                        continue

                    self.params_layer_list.append(param[1])
                    self.quant_layer_idx.append(i)
                    self.layer_type_info[i] = layer_name

        self.logger.info(self.layer_type_info)
        return

    def _get_quantable_layer_info(self):
        """
        params_size_list : params size of each quantizable layer (Conv2d , Linear)
        flops_size_list : flops size of each quantizable layer (Conv2d , Linear)
        """
        self.params_size_list = []
        self.flops_size_list = []

        m_list = list(self.model.modules())

        def new_forward(m):
            def lam_forward(x):
                m.input_feat = x.clone()
                self.measure_layer_for_quant(m, x)
                y = m.old_forward(x)
                m.output_feat = y.clone()
                return y

            return lam_forward

        for idx in self.quant_layer_idx:
            m = m_list[idx]
            m.old_forward = m.forward
            m.forward = new_forward(m)

        with torch.no_grad():
            for batch_id, data in enumerate(self.test_loader):

                _ = self.model(return_loss=False, **data)
                if batch_id == 0:
                    for idx in self.quant_layer_idx:
                        self.params_size_list.append(m_list[idx].params)
                        self.flops_size_list.append(m_list[idx].flops)
                else:
                    break
        return

    def _get_quantable_layer_weight_square(self, bit_width):

        quant_square_error = []

        for v in self.params_layer_list:
            if bit_width == 8:

                saturation_min = torch.min(v)
                saturation_max = torch.max(v)
                scale, zero_point = asymmetric_linear_quantization_params(bit_width, saturation_min, saturation_max)
                v_quant = asymmetric_quantize(v, bit_width, zero_point.to(self.device), specified_scale=scale)
                v_dequant = linear_dequantize(v_quant, scale, zero_point=zero_point.to(self.device), inplace=False)

                # square
                v_square = np.square(v.cpu().detach().numpy() - v_dequant.cpu().detach().numpy())
                v_error = np.sum(v_square)

            elif bit_width == 4:
                v_error = 0
                shape = v.size()

                for i in range(shape[0]):
                    for t in range(shape[1]):
                        saturation_min = torch.min(v[i][t])
                        saturation_max = torch.max(v[i][t])

                        scale, zero_point = asymmetric_linear_quantization_params(bit_width, saturation_min,
                                                                                  saturation_max)
                        v_quant = asymmetric_quantize(v[i][t], bit_width, zero_point, specified_scale=scale)
                        v_dequant = linear_dequantize(v_quant, scale, zero_point=zero_point, inplace=False)

                        v_square = np.square(v[i][t].cpu().detach().numpy() - v_dequant.cpu().detach().numpy())
                        v_error += np.sum(v_square)

            else:
                raise NotImplementedError('Just support 8-bit or 4-bit')

            quant_square_error.append(v_error)

        return quant_square_error

    def measure_layer_for_quant(self, layer, x):

        def get_layer_type(layer):
            layer_str = str(layer)
            return layer_str[:layer_str.find('(')].strip()

        def get_layer_param(model):
            params = 0
            for param in model.named_parameters():
                if 'bias' in param[0]:
                    continue
                if not param[1].requires_grad:
                    continue
                params += param[1].view(1, -1).shape[1]
            return params

        multi_add = 1
        type_name = get_layer_type(layer)

        # ops_conv
        if type_name in ['Conv2d']:
            out_h = int((x.size()[2] + 2 * layer.padding[0] - layer.kernel_size[0]) /
                        layer.stride[0] + 1)
            out_w = int((x.size()[3] + 2 * layer.padding[1] - layer.kernel_size[1]) /
                        layer.stride[1] + 1)
            layer.flops = layer.in_channels * layer.out_channels * layer.kernel_size[0] * \
                          layer.kernel_size[1] * out_h * out_w / layer.groups * multi_add
            layer.params = get_layer_param(layer)

        # ops_linear
        elif type_name in ['Linear']:

            weight_ops = layer.weight.numel() * multi_add
            bias_ops = layer.bias.numel()
            layer.flops = weight_ops + bias_ops
            layer.params = get_layer_param(layer)

        else:
            warnings.warn('Just support conv2d and linear!')

        return

    def _calibration(self):
        """
        Activation calibration on CPU
        """
        self.model.cpu()
        self.model.eval()

        quant_prepare(self.model)

        if self.cfg.target_type == 'detection':
            with torch.no_grad():
                for idx, data in enumerate(self.calibration_dataloader):
#                    _ = self.model(return_loss=False, **data)
                    if idx >= 500 - 1:
                        break

        elif self.cfg.target_type == 'classification':
            with torch.no_grad():
                for idx, data in enumerate(self.calibration_dataloader):
                    """inference and collect info"""
#                    output = self.model(**data)

                    if idx >= 500 - 1:
                        break
        else:
            raise Exception(f'Target_type is {self.cfg.target_type}')

        quant_convert(self.model)

        if self.model.__class__.__name__ == 'MMDataParallel':
            if self.model.module.__class__.__name__ == 'MMDataParallel':
                self.model = copy.deepcopy(self.model.module)

        return

    def put_model_on_gpus(self, model):
        if self.distributed:
            find_unused_parameters = self.cfg.get('find_unused_parameters', False)
            # Sets the `find_unused_parameters` parameter in
            # torch.nn.parallel.DistributedDataParallel
            model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
        else:
            if self.device == 'cpu':
                warnings.warn(
                    'The argument `device` is deprecated. To use cpu to train, '
                    'please refers to https://mmclassification.readthedocs.io/en'
                    '/latest/getting_started.html#train-a-model')
                model = model.cpu()
            else:
                model = MMDataParallel(model, device_ids=self.cfg.gpu_ids)
                if not model.device_ids:
                    from mmcv import __version__, digit_version
                    assert digit_version(__version__) >= (1, 4, 4), \
                        'To train with CPU, please confirm your mmcv version ' \
                        'is not lower than v1.4.4'
        return model

    def compute_bit_allocation(self):

        self.delta_weights_8bit_square = np.array(self._get_quantable_layer_weight_square(bit_width=8))
        self.delta_weights_4bit_square = np.array(self._get_quantable_layer_weight_square(bit_width=4))

        num_variable = self.params_size_list.shape[0]
        variable = {}

        for i in range(num_variable):
            variable[f"x{i}"] = LpVariable(f"x{i}", 1, 2, cat=LpInteger)

        prob = LpProblem("BitAllocation", LpMinimize)
        prob += sum(
            [0.5 * variable[f"x{i}"] * self.params_size_list[i] for i in range(num_variable)]) <= self.model_size_limit
        sensitivity_difference_between_4_8 = 1 * (self.delta_weights_8bit_square - self.delta_weights_4bit_square)
        prob += sum([(variable[f"x{i}"] - 1) * sensitivity_difference_between_4_8[i] for i in range(num_variable)])
        status = prob.solve(GLPK_CMD(msg=1, options=["--tmlim", "10000", "--simplex"]))
        LpStatus[status]

        result = []
        for i in range(num_variable):
            result.append(value(variable[f"x{i}"]))
        result = np.array(result)

        return result

    def quantization(self):

        self.q_model_params_size = 0
        self.BOPs = 0
        default_act_bitwidth = 8

        self._get_quantable_layer_index()
        self._get_quantable_layer_info()

        self.params_size_list = np.array(self.params_size_list) / 1024 / 1024
        self.quantable_params_size_32bit = np.sum(self.params_size_list) * 4.  # MB
        self.quantable_params_size_8bit = self.quantable_params_size_32bit / 4.  # 8bit model is 1/4 of 32bit model
        self.quantable_params_size_4bit = self.quantable_params_size_32bit / 8.  # 4bit model is 1/8 of 32bit model

        """For testing"""
        # self.cfg.quant_type = 'single'

        if self.cfg.quant_type == 'mix':
            # As mentioned previous, that's how we set the model size limit
            self.model_size_limit = self.quantable_params_size_4bit + \
                                    (self.quantable_params_size_8bit - self.quantable_params_size_4bit) * \
                                    self.cfg.model_size_limit_ratio
            # Bit width of each layer
            self.bit_width = self.compute_bit_allocation()
            self.logger.info('Bit width of each quantable layer: {} '.format(self.bit_width))

            for i in range(len(self.bit_width)):
                if self.bit_width[i] == 1:
                    # 4-bit
                    self.q_model_params_size += self.params_size_list[i] * 4. / 8.  # MB
                elif self.bit_width[i] == 2:
                    # 8-bit
                    self.q_model_params_size += self.params_size_list[i] * 4. / 4.  # MB
                else:
                    # 32-bit
                    self.q_model_params_size += self.params_size_list[i] * 4.  # MB
        else:
            # defualt quantization bit-width = 8
            self.bit_width = [2] * len(self.params_size_list)
            self.q_model_params_size = self.quantable_params_size_8bit

        self.q_model_params_size += self.org_params_size - self.quantable_params_size_32bit
        self.compression_ratio = (self.org_params_size - self.q_model_params_size) * 100.0 / self.org_params_size

        if not self.cfg.selfdistill_model:
            for i in range(len(self.bit_width)):
                self.BOPs += self.bit_width[i] * default_act_bitwidth * self.flops_size_list[i]
            self.BOPs = self.BOPs / 1e6

        i = 0
        m_list = list(self.model.modules())
        for idx in self.quant_layer_idx:
            m = m_list[idx]
            m.forward = m.old_forward

            state_dict = m.state_dict()
            for param in m.named_parameters():
                if 'bias' in param[0]:
                    continue
                if not param[1].requires_grad:
                    continue

                if self.bit_width[i] == 1:
                    bit = 4
                    i += 1
                    # channel-wise quantization
                    v = state_dict[param[0]]
                    shape = v.size()
                    v_dequant = copy.deepcopy(v)
                    for j in range(shape[0]):

                        for t in range(shape[1]):
                            saturation_min = torch.min(v[j][t])
                            saturation_max = torch.max(v[j][t])

                            scale, zero_point = asymmetric_linear_quantization_params(bit, saturation_min,
                                                                                      saturation_max)
                            v_quant = asymmetric_quantize(v[j][t], bit, zero_point.to(self.device),
                                                          specified_scale=scale)
                            v_dequant[j][t] = linear_dequantize(v_quant, scale, zero_point=zero_point.to(self.device),
                                                                inplace=False)

                elif self.bit_width[i] == 2:
                    bit = 8
                    i += 1
                    # layer-wise quantization 
                    v = state_dict[param[0]]
                    saturation_min = torch.min(v)
                    saturation_max = torch.max(v)

                    scale, zero_point = asymmetric_linear_quantization_params(bit, saturation_min, saturation_max)
                    v_quant = asymmetric_quantize(v, bit, zero_point.to(self.device), specified_scale=scale)
                    v_dequant = linear_dequantize(v_quant, scale, zero_point=zero_point.to(self.device), inplace=False)

                state_dict[param[0]] = v_dequant
                m.load_state_dict(state_dict)

        self.quant_acc = self._test_model(self.model)

        self.logger.info('{:<20s} :'.format('quant result'))
        self.logger.info('{:<20s} : {}'.format('Bit width of each quantable layer', self.bit_width))
        self.logger.info('{:<20s} : {:.2f}MB'.format('params size after quantized', self.q_model_params_size))

        if not self.cfg.selfdistill_model:
            self.logger.info('{:<20s} : {:.2f}M'.format('BOPs after quantized', self.BOPs))

        self.logger.info('{:<20s} : {}'.format('acc after quantized', self.quant_acc))
        self.logger.info('{:<20s} : {:.2f}%'.format('compression ratio', self.compression_ratio))

        self._save_result()

        return

    def _test_model(self, model):

        start = time.time()

        if not self.distributed:
            if self.cfg.target_type == 'classification':
                outputs = single_gpu_test_cls(model, self.test_loader)
            if self.cfg.target_type == 'detection':
                outputs = single_gpu_test_det(model, self.test_loader)
        else:
            raise NotImplementedError

        rank, _ = get_dist_info()
        if rank == 0:
            results = {}
            if self.cfg.evaluation.metric:

                if self.cfg.target_type == 'classification':
                    eval_results = self.test_loader.dataset.evaluate(
                        results=outputs,
                        metric=self.cfg.evaluation.metric,
                        logger=self.logger)

                    for k, v in eval_results.items():
                        if isinstance(v, np.ndarray):
                            v = [round(out, 2) for out in v.tolist()]
                            results[k] = v
                        elif isinstance(v, Number):
                            v = round(v, 2)
                            results[k] = v
                        else:
                            raise ValueError(f'Unsupport metric type: {type(v)}')

                if self.cfg.target_type == 'detection':
                    eval_results = self.test_loader.dataset.evaluate(
                        results=outputs,
                        metric=self.cfg.evaluation.metric,
                        logger=self.logger)

                    for k, v in eval_results.items():
                        if isinstance(v, np.ndarray):
                            v = [round(out, 4) * 100.0 for out in v.tolist()]
                            results[k] = v
                        elif isinstance(v, Number):
                            v = round(v, 4) * 100.0
                            results[k] = v
                        elif isinstance(v, str):
                            pass
                        else:
                            raise ValueError(f'Unsupport metric type: {type(v)}')

        end = time.time()
        time_ms = (end - start) * 1000

        if self.device == 'cpu':
            self.logger.info('{} {:<20s} : {:.4f}ms'.format('CPU', 'Inference Time', round(time_ms, 3)))
        else:
            self.logger.info('{} {:<20s} : {:.4f}ms'.format('GPU', 'Inference Time', round(time_ms, 3)))

        return results

    def _save_result(self):

        import codecs
        save_path = os.path.join(self.cfg.work_dir, self.cfg.timestamp + '_quant_result')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if self.model.__class__.__name__ == 'MMDataParallel':
            torch.save(self.model.module, os.path.join(save_path, 'quant_model.pth'))
        else:
            torch.save(self.model, os.path.join(save_path, 'quant_model.pth'))

        file_csv = codecs.open(os.path.join(save_path, 'quant_result.csv'), 'w+', 'utf-8')
        print(self.model, file=file_csv)
        print('{:<20s} :'.format('Bit width of each quantable layer'), file=file_csv)
        print(self.bit_width, file=file_csv)
        print('{:<20s} : {:.2f}MB'.format('original params size', self.org_params_size), file=file_csv)

        if not self.cfg.selfdistill_model:
            print('{:<20s} : {:.2f}M'.format('original BOPs', self.org_BOPs), file=file_csv)

        print('{:<20s} : {}'.format('original acc', self.org_acc), file=file_csv)

        print('{:<20s} :'.format('qaunt result'), file=file_csv)
        print('{:<20s} : {:.2f}MB'.format('params size after quantized', self.q_model_params_size), file=file_csv)

        if not self.cfg.selfdistill_model:
            print('{:<20s} : {:.2f}M'.format('BOPs after quantized', self.BOPs), file=file_csv)

        print('{:<20s} : {}'.format('acc after quantized', self.quant_acc), file=file_csv)
        print('{:<20s} : {:.2f}%'.format('compression ratio', self.compression_ratio), file=file_csv)
        file_csv.close()

        return

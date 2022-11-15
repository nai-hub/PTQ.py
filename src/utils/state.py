import torch
import torch.nn as nn
from torch.quantization.observer import HistogramObserver as HistogramObserver
from src.quant.core.quanter.linear_quanter import linear_quantize, linear_dequantize


def quant_prepare(model):
    for module in list(model.modules()):
        for name, mod in module._modules.items():

            if type(mod) == nn.ReLU or type(mod) == nn.ReLU6 or type(mod) == nn.LeakyReLU:
                setattr(module, name, nn.Sequential(*[mod, QuantizedAct(calibration_flag=True)]))
    return


def quant_convert(model):
    for module in model.modules():
        if type(module) == QuantizedAct:
            module.calibration_flag = False
    return


class QuantizedAct(nn.Module):
    def __init__(self, calibration_flag=True):
        super(QuantizedAct, self).__init__()

        self.calibration_flag = calibration_flag
        self.Histogram = HistogramObserver()
        self.scale = torch.ones(1)
        self.zero_point = torch.ones(1)
        self.act_quant_func = linear_quantize
        self.act_dequant_func = linear_dequantize

    def forward(self, x):
        if not self.calibration_flag:
            quant_act = self.act_quant_func(x, self.scale, self.zero_point)
            de_quant_act = self.act_dequant_func(quant_act, self.scale, self.zero_point)
            return de_quant_act
        else:
            x = self.Histogram(x)
            self.scale, self.zero_point = self.Histogram.calculate_qparams()
            return x

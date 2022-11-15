# Copyright (c) 2021 BeiHang University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
from torch import nn
from torch.autograd import Variable


def linear_quantize(input, scale, zero_point, inplace=False):
    """
    linear function : x_int = round(x/scale) + z_point
    """

    if len(input.shape) == 4:
        # reshape scale and zeropoint for convolutional weights and activations
        scale = scale.view(-1, 1, 1, 1)
        zero_point = zero_point.view(-1, 1, 1, 1)
    elif len(input.shape) == 2:
        # reshape scale and zeropoint for linear weights
        scale = scale.view(-1, 1)
        zero_point = zero_point.view(-1, 1)
    else:
        scale = scale.view(-1)
        zero_point = zero_point.view(-1)

    if inplace:
        input.mul_(1. / scale).add_(zero_point).round_()
        return input

    return torch.round(1. / scale * input + zero_point)


def linear_dequantize(input_q, scale, zero_point, inplace=False):
    """
    Map integer input tensor to fixed-point floating point with given scaling factor and zeropoint.

    Parameters:
    ----------
    input_q: quantized integer tensor to be mapped
    scale: scaling factor for quantization
    zero_pint: shift for quantization
    """

    if len(input_q.shape) == 4:
        # reshape scale and zeropoint for convolutional weights and activations
        scale = scale.view(-1, 1, 1, 1)
        zero_point = zero_point.view(-1, 1, 1, 1)

    elif len(input_q.shape) == 2:
        # reshape scale and zeropoint for linear weights
        scale = scale.view(-1, 1)
        zero_point = zero_point.view(-1, 1)
    else:
        scale = scale.view(-1)
        zero_point = zero_point.view(-1)

    if inplace:
        # mapping integer input_q to fixed-point floating point value with given scaling factor and zeropoint
        input_q.sub_(zero_point).mul_(scale)
        return input_q

    return (input_q - zero_point) * (scale)


def symmetric_linear_quantization_params(num_bits,
                                         saturation_min,
                                         saturation_max,
                                         per_channel=False):
    """
    Compute the scaling factor and zeropoint with the given quantization range for symmetric quantization.

    Parameters:
    ----------
    saturation_min: lower bound for quantization range
    saturation_max: upper bound for quantization range
    per_channel: if True, calculate the scaling factor per channel.
    """

    # these computation do not require any gradients, to enforce this, we use torch.no_grad()
    with torch.no_grad():
        n = 2 ** (num_bits - 1) - 1
        if per_channel:
            scale, _ = torch.max(torch.stack([saturation_min.abs(), saturation_max.abs()], dim=1), dim=1)
            scale = torch.clamp(scale, min=1e-8) / n
        else:
            scale = max(saturation_min.abs(), saturation_max.abs())
            scale = torch.clamp(scale, min=1e-8) / n

    return scale


def asymmetric_linear_quantization_params(num_bits,
                                          saturation_min,
                                          saturation_max,
                                          integral_zero_point=True):
    """
    Compute the scaling factor and zeropoint with the given quantization range for asymmetric quantization.

    Parameters:
    ----------
    saturation_min: lower bound for quantization range
    saturation_max: upper bound for quantization range
    integral_zero_point: if True, adjust zero_point accordingly to make sure 0.0 in floating point tensor
                         be exactly mapped to an integer value.
    """

    # these computation do not require any gradients, to enforce this, we use torch.no_grad()
    with torch.no_grad():
        n = 2 ** num_bits - 1
        scale = torch.clamp((saturation_max - saturation_min), min=1e-8) / float(n)

        # For asymmetric quantization, the current hardware support scaled unsigned integers without zero_point.
        # So saturation_min = 0 (we only do asymmetric quantization for activations after ReLU.)
        zero_point = -saturation_min / scale

        if integral_zero_point:
            if isinstance(zero_point, torch.Tensor):
                zero_point = zero_point.round()
            else:
                zero_point = float(round(zero_point))

        return scale, zero_point


def symmetric_quantize(input, bitwidth, zero_point, specified_scale=None):
    """
    x: floating point tensor to be quantized
    bitwidth: quantization bitwidth
    Note that the current implementation of SymmetricQuantFunction requires pre-calculated scaling factor.
    specified_scale: pre-calculated scaling factor for the tensor x
    
    """
    """ quantization range : [-n-1,n] """

    n = 2 ** (bitwidth - 1) - 1

    if specified_scale is not None:
        scale = specified_scale
    else:
        raise ValueError("The SymmetricQuantFunction requires a pre-calculated scaling factor")

    # if input.is_cuda == True:
    #     zero_point = torch.tensor(0.).cuda()
    # else:
    #     zero_point = torch.tensor(0.)
    # print(zero_point.device)

    new_quant_x = linear_quantize(input, scale, zero_point, inplace=False)
    new_quant_x = torch.clamp(new_quant_x, -n - 1, n)

    return new_quant_x


def asymmetric_quantize(input, bitwidth, zero_point, specified_scale=None):
    """
    x: floating point tensor to be quantized
    bitwidth: quantization bitwidth
    Note that the current implementation of SymmetricQuantFunction requires pre-calculated scaling factor.
    specified_scale: pre-calculated scaling factor for the tensor x
    
    """
    """ quantization range : [0,n] """
    n = 2 ** (bitwidth) - 1

    """ quantization range : [-n-1,n] """
    # n = 2 ** (bitwidth-1) - 1

    if specified_scale is not None:
        scale = specified_scale
    else:
        raise ValueError("The SymmetricQuantFunction requires a pre-calculated scaling factor")

    # if input.is_cuda == True:
    #     zero_point = torch.tensor(0.).cuda()
    # else:
    #     zero_point = torch.tensor(0.)
    # print(zero_point.device)

    new_quant_x = linear_quantize(input, scale, zero_point, inplace=False)
    new_quant_x = torch.clamp(new_quant_x, 0, n)
    # new_quant_x = torch.clamp(new_quant_x, -n-1, n) 

    return new_quant_x


"""-------------------------------------------------------------------------"""

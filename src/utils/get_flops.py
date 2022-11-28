try:
    from mmcv.cnn import get_model_complexity_info
except ImportError:
    raise ImportError('Please upgrade mmcv to >0.6.2')
    
import torch
import copy

    
def get_flops(model,input_shape,as_strings=False):
    
    assert isinstance(input_shape,list)
    
    if len(input_shape) == 1:
        h = w = input_shape[0]
    elif len(input_shape) == 2:
        h, w = input_shape
    else:
        raise ValueError('invalid input shape')
        
    dump_input = (3,h,w)
    
    if model.__class__.__name__ == 'MMDataParallel':
        m = copy.deepcopy(model.module)
    else:
        m = copy.deepcopy(model)
    
    if hasattr(m, 'extract_feat'):
        m.forward = m.extract_feat
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
            format(m.__class__.__name__))
    
    flops,_ = get_model_complexity_info(m, dump_input,as_strings=as_strings)
    
    return flops


    
    
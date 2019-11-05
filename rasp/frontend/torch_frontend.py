# -*- coding: utf-8 -*-
import sys
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from rasp.profiler.tree import StatTreeNode
from rasp.profiler.hook import Tape, hook_compute_in, hook_compute_out,\
                               hook_time_start, hook_time_stop
from rasp.utils.time import Timer, get_cpu_time, get_time

def get_num_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def get_dtype(module):
    pass

def get_current_mem():
    return torch.cuda.memory_allocated()

def synchronize():
    torch.cuda.synchronize()

def reg_conv(m):
    return {
        'C_in': m.in_channels,
        'C_out': m.out_channels,
        'kernel_size': m.kernel_size,
        'stride': m.stride,
        'padding': m.padding,
        'bias': False if m.bias is None else True,
        'dilation': m.dilation,
        'groups': m.groups,
    }

def reg_pool(m):
    typestr = str(type(m).__name__)
    ptype = 'avg' if typestr.find('Max') == -1 else 'max'
    adapt = False if typestr.find('Adaptive') == -1 else True
    return {
        'kernel_size': 0 if adapt else m.kernel_size,
        'stride': 0 if adapt else m.stride,
        'padding': 0 if adapt else m.padding,
        'pool_type': ptype,
        'adaptive': adapt,
    }

def reg_bn(m):
    return {
        'num_feat': m.num_features,
        'affine': m.affine,
        'running_stats': m.track_running_stats,
    }

def reg_act(m):
    return {
        'inplace': m.inplace,
    }

def reg_linear(m):
    return {
        'in_feat': m.in_features,
        'out_feat': m.out_features,
        'bias': False if m.bias is None else True,
    }

def reg_upsample(m):
    return {
        'C_in': m.in_channels,
        'C_out': m.out_channels,
        'mode': m.mode
    }

def reg_stat(node, module):
    stdtype = node['stdtype']
    if stdtype == 'CONV':
        m_stats = reg_conv(module)
    elif stdtype == 'BN':
        m_stats = reg_bn(module)
    elif stdtype == 'POOL':
        m_stats = reg_pool(module)
    elif stdtype == 'ACT':
        m_stats = reg_act(module)
    elif stdtype == 'UPSMPL':
        m_stats = reg_upsample(module)
    elif stdtype == 'FC':
        m_stats = reg_linear(module)
    elif stdtype == 'IDT':
        m_stats = {}
    else:
        # print(f"reg {type(module).__name__} is not supported!")
        m_stats = {}
    node.update_values(m_stats)

_stdtype_map = {
    nn.Conv1d: 'CONV',
    nn.Conv2d: 'CONV',
    nn.Conv3d: 'CONV',
    nn.ConvTranspose1d: 'CONV',
    nn.ConvTranspose2d: 'CONV',
    nn.ConvTranspose3d: 'CONV',

    nn.BatchNorm1d: 'BN',
    nn.BatchNorm2d: 'BN',
    nn.BatchNorm3d: 'BN',

    nn.ReLU: 'ACT',
    nn.ReLU6: 'ACT',
    nn.PReLU: 'ACT',
    nn.ELU: 'ACT',
    nn.LeakyReLU: 'ACT',

    nn.MaxPool1d: 'POOL',
    nn.MaxPool2d: 'POOL',
    nn.MaxPool3d: 'POOL',
    nn.AdaptiveMaxPool1d: 'POOL',
    nn.AdaptiveMaxPool2d: 'POOL',
    nn.AdaptiveMaxPool3d: 'POOL',

    nn.AvgPool1d: 'POOL',
    nn.AvgPool2d: 'POOL',
    nn.AvgPool3d: 'POOL',
    nn.AdaptiveAvgPool1d: 'POOL',
    nn.AdaptiveAvgPool2d: 'POOL',
    nn.AdaptiveAvgPool3d: 'POOL',

    nn.Linear: 'FC',
    nn.Dropout: 'NONE',

    nn.Upsample: 'UPSMPL',
    nn.UpsamplingBilinear2d: 'UPSMPL',
    nn.UpsamplingNearest2d: 'UPSMPL',
}

def get_stdtype(module):
    typ = type(module)
    if typ in _stdtype_map:
        return _stdtype_map[typ]
    if len(list(module.named_children()))==0:
        sys.stderr.write('RASP: torch frontend: unrecognized module: {}\n'.format(str(typ)))
    return 'NONE'

def reg_stats_node(m, prefix=''):
    if hasattr(m, '_RASPStatNode'): return m._RASPStatNode
    node = StatTreeNode(prefix)
    node['type'] = type(m).__name__
    node['stdtype'] = get_stdtype(m)
    node['name'] = prefix
    node['params'] = get_num_params(m)
    h_in = m.register_forward_pre_hook(hook_module_in)
    h_out = m.register_forward_hook(hook_module_out)
    node.tape = Tape(node=node)
    node.module = m
    node.hooks = [h_in, h_out]
    m._RASPStatNode = node
    if len(list(m.named_children()))==0:
        reg_stat(node, m)
    else:
        for sn, sm in m.named_children():
            node.add_child(sn, reg_stats_node(sm, prefix+'.'+sn))
    return node

def unreg_stats_node(m):
    if not hasattr(m, '_RASPStatNode'): return
    node = m._RASPStatNode
    node.tape.clear()
    del node.tape
    for h in node.hooks:
        h.remove()
    for sn, sm in m.named_children():
        unreg_stats_node(sm)
    del m._RASPStatNode

def hook_module_in(module, input):
    t0 = get_cpu_time()
    node = module._RASPStatNode
    node['dev_mem'] = get_current_mem()
    tape = node.tape
    tape.clear()
    tape.reg_parent('tape')
    if node['hook_comp']:
        hook_compute_in(node, tuple(input[0].shape))
    if node['hook_time']:
        hook_time_start(node, t0)

def hook_module_out(module, input, output):
    t0 = get_cpu_time()
    node = module._RASPStatNode
    node['dev_mem_alloc'] = get_current_mem() - node['dev_mem']
    node['fwd'] = 1 + (node['fwd'] or 0)
    if node['hook_comp']:
        hook_compute_out(node, tuple(input[0].shape), tuple(output.shape))
    if node['hook_time']:
        hook_time_stop(node, t0)

_origin_call = dict()
def wrap_call(module, *input, **kwargs):
    hook_module_in(module, input)
    output = _origin_call[module.__class__](module, *input, **kwargs)
    hook_module_out(module, input)
    return output

def hook_call(module, max_depth=-1):
    if max_depth == 0: return
    node = module._RASPStatNode
    if not module.__class__.__call__ is wrap_call:
        global _origin_call
        _origin_call[module.__class__] = module.__class__.__call__
        module.__class__.__call__ = wrap_call
    for n, m in module.named_children():
        hook_call(m, max_depth-1)

def unhook_call(module):
    node = module._RASPStatNode
    if module.__class__.__call__ is wrap_call:
        module.__class__.__call__ = _origin_call[module.__class__]
    for n, m in module.named_children():
        unhook_call(m)

def hook_timing(module, max_depth=-1):
    if max_depth == 0: return
    node = module._RASPStatNode
    if node['hook_time']: return
    node['timer'] = Timer(time_src=get_cpu_time, synch=synchronize)
    node['net_timer'] = Timer(time_src=get_cpu_time, synch=synchronize)
    node['hook_time'] = True
    for n, m in module.named_children():
        hook_timing(m, max_depth-1)

def unhook_timing(module):
    if not hasattr(module, '_RASPStatNode'): return
    node = module._RASPStatNode
    if not node['hook_time']: return
    node['hook_time'] = False
    for n, m in module.named_children():
        unhook_timing(m)

def hook_compute(module, max_depth=-1):
    if max_depth == 0: return
    node = module._RASPStatNode
    if node['hook_comp']: return
    node['hook_comp'] = True
    for n, m in module.named_children():
        hook_compute(m, max_depth-1)

def unhook_compute(module):
    if not hasattr(module, '_RASPStatNode'): return
    node = module._RASPStatNode
    if not node['hook_comp']: return
    node['hook_comp'] = False
    for n, m in module.named_children():
        unhook_compute(m)

def run(module, inputs):
    with torch.no_grad():
        return module(inputs)

def pre_run(module, inputs):
    module.eval()
    return module

def get_random_data(data_size):
    data = torch.rand(data_size)
    return data
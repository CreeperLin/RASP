# -*- coding: utf-8 -*-
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
    return {
        'kernel_size': m.kernel_size,
        'stride': m.stride,
        'padding': m.padding,
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
        'kernel_size': m.kernel_size,
        'stride': m.stride,
        'padding': m.padding,
        'bias': False if m.bias is None else True,
        'dilation': m.dilation,
        'groups': m.groups,
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


def get_stdtype(module):
    if isinstance(module, nn.Conv2d):
        return 'CONV'
    elif isinstance(module, nn.BatchNorm2d):
        return 'BN'
    elif isinstance(module, (nn.AvgPool2d, nn.MaxPool2d)):
        return 'POOL'
    elif isinstance(module, (nn.ReLU, nn.ReLU6, nn.PReLU, nn.ELU, nn.LeakyReLU)):
        return 'ACT'
    elif isinstance(module, nn.Upsample):
        return 'UPSMPL'
    elif isinstance(module, nn.Linear):
        return 'FC'
    elif isinstance(module, nn.Identity):
        return 'IDT'
    else:
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
    node.tape = Tape(node=node, is_set=False)
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
    for h in node.hooks:
        h.remove()
    for sn, sm in m.named_children():
        unreg_stats_node(sm)

def hook_module_in(module, input):
    t0 = get_cpu_time()
    node = module._RASPStatNode
    tape = node.tape
    tape.clear()
    tape.reg_parent('tape')
    if node['hook_comp']:
        hook_compute_in(node, input[0].shape)
    if node['hook_time']:
        hook_time_start(node, t0)

def hook_module_out(module, input, output):
    t0 = get_cpu_time()
    node = module._RASPStatNode
    node['fwd'] = 1 + (node['fwd'] or 0)
    if node['hook_comp']:
        hook_compute_out(node, input[0].shape, output.shape)
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
    node['timer'] = Timer(time_src=get_cpu_time)
    node['net_timer'] = Timer(time_src=get_cpu_time)
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

def run(module, X):
    with torch.no_grad():
        return module(X)

def pre_run(module, X):
    module.eval()
    return module

def get_random_data(data_size):
    data = torch.rand(data_size)
    return data
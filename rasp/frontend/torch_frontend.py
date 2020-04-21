# -*- coding: utf-8 -*-
import os
import logging
import torch
import torch.nn as nn
from ..profiler.tree import StatTreeNode
from ..profiler.hook import Tape, hook_compute_in, hook_compute_out,\
                               hook_time_start, hook_time_stop
from ..utils.time import Timer, get_cpu_time, get_time
from .. import device as DEV
from ..utils.config import CFG

logger = logging.getLogger('rasp')

def init():
    pass

def get_num_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def get_dtype(module):
    p_list = list(module.parameters())
    if len(p_list) == 0: return None
    return str(p_list[0].dtype)[6:]

def get_data_shape(data):
    if isinstance(data, (tuple, list)):
        ret = [get_data_shape(d) for d in data]
        if len(ret) == 1:
            ret = ret[0]
        return ret
    elif isinstance(data, torch.Tensor):
        return tuple(data.shape)

def get_device(module):
    p_list = list(module.parameters())
    if len(p_list) == 0: return None
    return p_list[0].device

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
        logger.debug('torch frontend: unrecognized module: {}\n'.format(str(typ)))
    return 'NONE'

def get_stats_node(m, input_shape=None):
    if hasattr(m, '_RASPStatNodes'): return m._RASPStatNodes.get(str(input_shape), None)
    return None 

def get_stats_node_all(m):
    if hasattr(m, '_RASPStatNodes'): return m._RASPStatNodes
    return None 

def set_stats_node(m, node, input_shape=None):
    if not hasattr(m, '_RASPStatNodes'):
        m._RASPStatNodes = {}
    m._RASPStatNodes[str(input_shape)] = node

def unset_stats_node(m, input_shape=None):
    if hasattr(m, '_RASPStatNodes'):
        del m._RASPStatNodes[str(input_shape)]
        if len(m) == 0:
            del m._RASPStatNodes

def unset_stats_node_all(m):
    if hasattr(m, '_RASPStatNodes'):
        del m._RASPStatNodes

def build_stats_node(m, prefix, hook, tape):
    node = StatTreeNode(prefix)
    node.module = m
    node['type'] = type(m).__name__
    node['stdtype'] = get_stdtype(m)
    node['name'] = prefix
    node['params'] = get_num_params(m)
    node['dtype'] = get_dtype(m)
    node.tape = Tape(node=node) if tape else None
    node.hooks = []
    if hook:
        h_in = m.register_forward_pre_hook(hook_module_in)
        h_out = m.register_forward_hook(hook_module_out)
        node.hooks = [h_in, h_out]
    return node

def reg_stats_node(m, prefix='', hook=True, tape=False):
    node = get_stats_node(m)
    if not node is None:
        return node
    node = build_stats_node(m, prefix, hook, tape)
    set_stats_node(m, node)
    reg_stat(node, m)
    for sn, sm in m.named_children():
        node.add_child(sn, reg_stats_node(sm, prefix+'.'+sn))
    return node

def unreg_stats_node(m):
    nodes = get_stats_node_all(m)
    if nodes is None: return
    for node in nodes.values():
        for h in node.hooks:
            h.remove()
    for sn, sm in m.named_children():
        unreg_stats_node(sm)
    unset_stats_node_all(m)

def hook_module_in(module, input):
    t0 = get_cpu_time()
    nodes_all = get_stats_node_all(module)
    default_node = get_stats_node(module)
    input_shape = get_data_shape(input)
    cur_node = get_stats_node(module, input_shape)
    if len(nodes_all) == 1:
        set_stats_node(module, default_node, input_shape)
        default_node.tape = Tape(node=default_node)
        cur_node = default_node
    elif cur_node is None:
        cur_node = build_stats_node(module, default_node.name, hook=False, tape=True)
        cur_node.parent = default_node.parent
        cur_node._children = default_node._children
        reg_stat(cur_node, module)
        set_stats_node(module, cur_node, input_shape)
    if CFG.frontend.stat_mem:
        cur_node['dev_mem'] = DEV.get_current_mem()
        cur_node['net_dev_mem'] = cur_node['dev_mem'] - DEV.get_base_mem()
    if default_node['hook_comp']:
        tape = cur_node.tape
        default_node.tape = tape
        if not tape is None:
            tape.clear()
            tape.reg_parent()
        hook_compute_in(cur_node, input_shape)
    if default_node['hook_time']:
        if cur_node['timer'] is None:
            cur_node['timer'] = Timer(time_src=get_cpu_time, synch=DEV.get_synchronize())
            cur_node['net_timer'] = Timer(time_src=get_cpu_time, synch=DEV.get_synchronize())
        hook_time_start(cur_node, t0)

def hook_module_out(module, input, output):
    t0 = get_cpu_time()
    default_node = get_stats_node(module)
    input_shape = get_data_shape(input)
    cur_node = get_stats_node(module, input_shape)
    assert cur_node is not None
    if CFG.frontend.stat_mem:
        cur_node['dev_mem_alloc'] = DEV.get_current_mem() - cur_node['dev_mem']
        cur_node['dev_max_mem'] = DEV.get_max_mem()
        cur_node['dev_max_mem_alloc'] = cur_node['dev_max_mem'] - DEV.get_base_mem()
        if CFG.frontend.reset_max_mem:
            DEV.reset_max_mem()
    cur_node['fwd'] = 1 + (cur_node['fwd'] or 0)
    if default_node['hook_comp']:
        hook_compute_out(cur_node, input_shape, get_data_shape(output), CFG.frontend.mark_updated)
    if default_node['hook_time']:
        hook_time_stop(cur_node, t0)
    default_node.stats.update(cur_node.stats)
    DEV.add_node(cur_node)

_origin_call = dict()
def wrap_call(module, *input, **kwargs):
    hook_module_in(module, input)
    output = _origin_call[module.__class__](module, *input, **kwargs)
    hook_module_out(module, input)
    return output

def hook_call(module, max_depth=-1):
    if max_depth == 0: return
    node = get_stats_node(module)
    if not module.__class__.__call__ is wrap_call:
        global _origin_call
        _origin_call[module.__class__] = module.__class__.__call__
        module.__class__.__call__ = wrap_call
    for n, m in module.named_children():
        hook_call(m, max_depth-1)

def unhook_call(module):
    node = get_stats_node(module)
    if module.__class__.__call__ is wrap_call:
        module.__class__.__call__ = _origin_call[module.__class__]
    for n, m in module.named_children():
        unhook_call(m)

def hook_timing(module, max_depth=-1):
    if max_depth == 0: return
    node = get_stats_node(module)
    if node['hook_time']: return
    node['hook_time'] = True
    for n, m in module.named_children():
        hook_timing(m, max_depth-1)

def unhook_timing(module):
    node = get_stats_node(module)
    if node is None: return
    if not node['hook_time']: return
    node['hook_time'] = False
    for n, m in module.named_children():
        unhook_timing(m)

def hook_compute(module, max_depth=-1):
    if max_depth == 0: return
    node = get_stats_node(module)
    if node['hook_comp']: return
    node['hook_comp'] = True
    for n, m in module.named_children():
        hook_compute(m, max_depth-1)

def unhook_compute(module):
    node = get_stats_node(module)
    if node is None: return
    if not node['hook_comp']: return
    node['hook_comp'] = False
    for n, m in module.named_children():
        unhook_compute(m)


def get_random_data(data_size):
    data = torch.rand(data_size)
    return data


class profile_ctx():
    def __init__(self, module, inputs, device):
        self.module = module
        self.inputs = inputs
        self.last_device = get_device(module)
        self.device = self.last_device
        self.module_training = module.training
        if not device is None:
            self.device = device
    
    def __enter__(self):
        self.module.eval()
        self.module.to(self.device)
        self.inputs = self.inputs.to(self.device)
        DEV.reset()
        return self

    def run(self):
        with torch.no_grad():
            return self.module(self.inputs)

    def __exit__(self, type, value, trace):
        if not self.last_device is None:
            self.module.to(self.last_device)
        self.module.train(self.module_training)


def get_ctx(module, inputs, device):
    return profile_ctx(module, inputs, device)


def run(module, inputs, device):
    with profile_ctx(module, inputs, device) as ctx:
        ctx.run()

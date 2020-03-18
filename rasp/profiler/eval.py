# -*- coding: utf-8 -*-
import numpy as np

def eval_conv(node):
    in_c = node['C_in']
    out_c = node['C_out']
    params = node['params']
    kernel_size = node['kernel_size']
    sh, sw = node['stride']
    ph, pw = node['padding']
    bias = node['bias']
    groups = node['groups']
    in_shape = node['in_shape']
    out_shape = node['out_shape']
    bs, _, ih, iw = in_shape
    kh, kw = kernel_size[:2]
    oh, ow = (ih + 2 * ph - kh) / sh + 1, (iw + 2 * pw - kw) / sw + 1
    out_feat = oh * ow
    out_numel = out_c * out_feat
    in_numel = in_c * ih * iw
    
    kernel_numel = int(np.prod(kernel_size))
    bias_ops = 1 if bias else 0
    kernel_ops = kernel_numel

    flops = bs * out_numel * (in_c // groups * kernel_ops - 1 + bias_ops)

    mem_r = bs * (in_numel + params)
    mem_w = bs * out_numel

    return {
        'flops': int(flops),
        'mem_r': mem_r,
        'mem_w': mem_w,
    }

def eval_bn(node):
    params = node['params']
    affine = node['affine']
    running_stats = node['running_stats']
    in_shape = node['in_shape']
    num_feat = node['num_feat']
    bs = in_shape[0]
    in_numel = int(np.prod(in_shape[1:]))
    
    affine_ops = 1 if affine else 0

    flops = bs * (affine_ops + 3) * in_numel

    mem_r = bs * (in_numel + params)
    mem_w = bs * in_numel

    return {
        'flops': int(flops),
        'mem_r': mem_r,
        'mem_w': mem_w,
    }

def eval_pool(node):
    ptype = node['pool_type']
    adapt = node['adaptive']
    params = node['params']
    in_shape = node['in_shape']
    out_shape = node['out_shape']
    in_feat = in_shape[2:]
    if adapt:
        in_shape = np.array(in_shape)
        out_shape = np.array(out_shape)
        s = in_shape // out_shape
        k = in_shape - (out_shape-1) * s
        p = 0
    else:
        k = node['kernel_size']
        s = node['stride']
        p = node['padding']
    kernel_numel = int(np.prod(k))

    kernel_mul = 1 if ptype == 'avg' else 0
    kernel_add = kernel_numel - 1
    kernel_ops = kernel_mul + kernel_add

    bs = out_shape[0]
    out_numel = int(np.prod(out_shape[1:]))
    in_numel = int(np.prod(in_shape[1:]))
    
    flops = bs * kernel_ops * out_numel

    mem_r = bs * (in_numel + params)
    mem_w = bs * out_numel

    return {
        'flops': int(flops),
        'mem_r': mem_r,
        'mem_w': mem_w,
    }

def eval_act(node):
    params = node['params']
    in_shape = node['in_shape']
    bs = in_shape[0]
    num_feat = in_shape[1:]
    in_numel = int(np.prod(num_feat))

    flops = mem_r = mem_w = bs * in_numel 

    return {
        'flops': int(flops),
        'mem_r': mem_r,
        'mem_w': mem_w,
    }

def eval_upsample(node):
    C_in = node['C_in']
    C_out = node['C_out']
    params = node['params']
    in_shape = node['in_shape']
    out_shape = node['out_shape']
    mode = node['mode']

    bs = out_shape[0:]
    out_feat = out_shape[1:]
    out_numel = int(np.prod(out_feat))
    in_numel = int(np.prod(in_shape[1:]))

    if mode == "nearest":
        flops = 0
    if mode == "linear":
        flops = out_numel * 5 # 2 muls + 3 add
    elif mode == "bilinear":
        flops = out_numel * 13 # 6 muls + 7 adds
    elif mode == "bicubic":
        ops_solve_A = 224 # 128 muls + 96 adds
        ops_solve_p = 35 # 16 muls + 12 adds + 4 muls + 3 adds
        ops = (ops_solve_A + ops_solve_p)
        flops = out_numel * ops
    elif mode == "trilinear":
        ops = (13 * 2 + 5)
        flops = out_numel * ops
    flops = bs * flops

    mem_r = bs * (in_numel + params)
    mem_w = bs * out_numel

    return {
        'flops': int(flops),
        'mem_r': mem_r,
        'mem_w': mem_w,
    }

def eval_linear(node):
    in_feat = node['in_feat']
    out_feat = node['out_feat']
    params = node['params']
    bias = node['bias']
    in_shape = node['in_shape']
    bs = in_shape[0]
    
    bias_ops = 1 if bias else 0

    flops = bs * out_feat * (in_feat - 1 + bias_ops)

    mem_r = bs * (in_feat + params)
    mem_w = bs * out_feat

    return {
        'flops': int(flops),
        'mem_r': mem_r,
        'mem_w': mem_w,
    }

def eval_identity(node):
    return {
        'flops': 0,
        'mem_r': 0,
        'mem_w': 0,
    }

def eval_nullop(node):
    return {
        'flops': 0,
        'mem_r': 0,
        'mem_w': 0,
    }


_evaluators = {}
_evaluators_all = []


def add_evaluator(ntype, func):
    if ntype == 'all':
        _evaluators_all.append(func)
        return
    if not isinstance(ntype, list):
        ntype = [ntype]
    for typ in ntype:
        evals = _evaluators.get(typ, None)
        if evals is None:
            evals = []
            _evaluators[typ] = evals
        evals.append(func)


def get_evaluators(ntype, default=None):
    if ntype == 'all':
        return _evaluators_all
    evals = _evaluators.get(ntype, None)
    if evals is None:
        return [] if default is None else [default]
    return evals


def eval_compute_prop(node, mark_updated=True):
    if node['compute_updated']: return
    assert not node['in_shape'] is None
    ntype = node['type']
    stdtype = node['stdtype']
    evals = get_evaluators(stdtype, eval_nullop)
    evals.extend(get_evaluators(ntype))
    evals.extend(get_evaluators('all'))
    for eval_fn in evals:
        node.update_values(eval_fn(node))
    if mark_updated:
        node['compute_updated'] = True


def eval_compute_nofwd(node, in_shape=None, out_shape=None):
    out_shape = in_shape if out_shape is None else out_shape
    node['in_shape'] = in_shape
    node['out_shape'] = out_shape
    if node.num_children == 0:
        eval_compute_prop(node, False)
    else:
        node['flops'] = 0
        node['mem_r'] = 0
        node['mem_w'] = 0
        n_in, n_out = in_shape, out_shape
        for n in node.children:
            eval_compute_nofwd(n, n_in, n_out)
            # n_in, n_out = n_out
            node['flops'] += n['flops']
            node['mem_r'] += n['mem_r']
            node['mem_w'] += n['mem_w']


add_evaluator('CONV', eval_conv)
add_evaluator('BN', eval_bn)
add_evaluator('POOL', eval_pool)
add_evaluator('IDT', eval_identity)
add_evaluator('FC', eval_linear)
add_evaluator('ACT', eval_act)
add_evaluator('UPSMPL', eval_upsample)

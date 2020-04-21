# -*- coding: utf-8 -*-

def prod(arr):
    if not isinstance(arr, (tuple, list)):
        return arr
    ret = 1
    for i in arr:
        ret *= i
    return ret

def eval_conv(node):
    kernel_size = node['kernel_size']
    sh, sw = node['stride']
    ph, pw = node['padding']
    bias = node['bias']
    groups = node['groups']
    in_shape = node['in_shape']
    out_shape = node['out_shape']
    bs, in_c, ih, iw = in_shape
    _, out_c, oh, ow = out_shape
    out_numel = out_c * oh * ow
    in_numel = in_c * ih * iw
    
    kernel_numel = prod(kernel_size)
    bias_ops = 1 if bias else 0
    kernel_ops = kernel_numel
    params = out_c * in_c * kernel_numel // groups + out_c * bias_ops

    kernel_macc = in_c // groups * kernel_ops
    flops = bs * out_numel * (kernel_macc * 2 - 1 + bias_ops)
    macc = bs * out_numel * (kernel_macc + bias_ops)

    mem_r = bs * (in_numel + params)
    mem_w = bs * out_numel

    return {
        'flops': flops,
        'macc': macc,
        'mem_r': mem_r,
        'mem_w': mem_w,
        'params': params,
    }

def eval_bn(node):
    affine = node['affine']
    running_stats = node['running_stats']
    in_shape = node['in_shape']
    num_feat = node['num_feat']
    bs = in_shape[0]
    in_numel = prod(in_shape[1:])
    
    affine_ops = 1 if affine else 0

    macc = bs * (affine_ops + 1) * in_numel
    flops = 2 * macc
    params = 2 * affine_ops * in_shape[1]

    mem_r = bs * (in_numel + params)
    mem_w = bs * in_numel

    return {
        'flops': flops,
        'macc': macc,
        'mem_r': mem_r,
        'mem_w': mem_w,
        'params': params
    }

def eval_pool(node):
    ptype = node['pool_type']
    adapt = node['adaptive']
    params = node['params']
    in_shape = node['in_shape']
    out_shape = node['out_shape']
    in_feat = in_shape[2:]
    if adapt:
        k = [in_d - (out_d-1) * in_d // out_d for in_d, out_d in zip(in_shape, out_shape)]
        p = 0
    else:
        k = node['kernel_size']
        s = node['stride']
        p = node['padding']
    kernel_numel = prod(k)

    kernel_mul = kernel_numel if ptype == 'avg' else 0
    kernel_add = kernel_numel - 1
    kernel_ops = kernel_mul + kernel_add

    bs = out_shape[0]
    out_numel = prod(out_shape[1:])
    in_numel = prod(in_shape[1:])
    
    flops = bs * kernel_ops * out_numel
    macc = bs * kernel_numel * out_numel

    mem_r = bs * (in_numel + params)
    mem_w = bs * out_numel

    return {
        'flops': flops,
        'macc': macc,
        'mem_r': mem_r,
        'mem_w': mem_w,
    }

def eval_act(node):
    params = node['params']
    in_shape = node['in_shape']
    bs = in_shape[0]
    num_feat = in_shape[1:]
    in_numel = prod(num_feat)

    macc = flops = mem_r = mem_w = bs * in_numel

    return {
        'flops': flops,
        'macc': macc,
        'mem_r': mem_r,
        'mem_w': mem_w,
    }

def eval_upsample(node):
    params = node['params']
    in_shape = node['in_shape']
    out_shape = node['out_shape']
    mode = node['mode']

    bs = out_shape[0:]
    out_feat = out_shape[1:]
    out_numel = prod(out_feat)
    in_numel = prod(in_shape[1:])

    if mode == "nearest":
        flops = 0
    if mode == "linear":
        flops = out_numel * 5 # 2 muls + 3 add
        macc = out_numel * 3
    elif mode == "bilinear":
        flops = out_numel * 13 # 6 muls + 7 adds
        macc = out_numel * 7
    elif mode == "bicubic":
        ops_solve_A = 224 # 128 muls + 96 adds
        ops_solve_p = 35 # 16 muls + 12 adds + 4 muls + 3 adds
        ops = (ops_solve_A + ops_solve_p)
        flops = out_numel * ops
        macc = out_numel * 148
    elif mode == "trilinear":
        ops = (13 * 2 + 5)
        flops = out_numel * ops
        macc = out_numel * 13 * 2
    flops = bs * flops

    mem_r = bs * (in_numel + params)
    mem_w = bs * out_numel

    return {
        'flops': flops,
        'macc': macc,
        'mem_r': mem_r,
        'mem_w': mem_w,
    }

def eval_linear(node):
    in_feat = node['in_feat']
    out_feat = node['out_feat']
    bias = node['bias']
    in_shape = node['in_shape']
    bs = in_shape[0]
    
    bias_ops = 1 if bias else 0

    flops = bs * out_feat * (in_feat * 2 - 1 + bias_ops)
    macc = bs* out_feat * (in_feat + bias_ops)
    params = out_feat * (in_feat + bias_ops)

    mem_r = bs * (in_feat + params)
    mem_w = bs * out_feat

    return {
        'flops': flops,
        'macc': macc,
        'mem_r': mem_r,
        'mem_w': mem_w,
        'params': params
    }

def eval_identity(node):
    return {
        'flops': 0,
        'macc': 0,
        'mem_r': 0,
        'mem_w': 0,
    }

def eval_nullop(node):
    return {
        'flops': 0,
        'macc': 0,
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


def eval_compute_prop(node):
    if node['compute_updated']: return
    assert not node['in_shape'] is None
    ntype = node['type']
    stdtype = node['stdtype']
    evals = get_evaluators(stdtype, eval_nullop)
    evals.extend(get_evaluators(ntype))
    evals.extend(get_evaluators('all'))
    for eval_fn in evals:
        node.update_values(eval_fn(node))


def eval_compute_nofwd(node, in_shape=None, out_shape=None):
    out_shape = in_shape if out_shape is None else out_shape
    node['in_shape'] = in_shape
    node['out_shape'] = out_shape
    if node.num_children == 0:
        eval_compute_prop(node)
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

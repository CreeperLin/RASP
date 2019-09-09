# -*- coding: utf-8 -*-
import numpy as np

def numel_to_bytes(numel, prec='float'):
    if prec == 'double':
        return numel * 8
    elif prec == 'half':
        return numel * 2
    elif prec == 'float':
        return numel * 4
    else:
        return numel * 4

def eval_conv(node):
    C_in = node['C_in']
    C_out = node['C_out']
    params = node['params']
    kernel_size = node['kernel_size']
    s = node['stride']
    p = node['padding']
    bias = node['bias']
    groups = node['groups']
    in_shape = node['in_shape']
    out_shape = node['out_shape']
    bs, out_c, out_h, out_w = out_shape
    in_c, in_h, in_w = in_shape[1:]
    out_numel = out_c * out_h * out_w
    
    kernel_numel = np.prod(kernel_size)
    bias_ops = 1 if bias else 0
    kernel_mul = kernel_numel * (in_c // groups)
    kernel_add = kernel_mul - 1 + bias_ops
    kernel_mul_group = kernel_mul * out_h * out_w * (out_c // groups)
    kernel_add_group = kernel_add * out_h * out_w * (out_c // groups)
    total_mul = kernel_mul_group * groups
    total_add = kernel_add_group * groups
    madds = total_mul + total_add

    kernel_ops = kernel_numel + bias_ops
    flops = bs * out_numel * (in_c // groups * kernel_ops + bias_ops)

    mem_r = bs * (out_numel + params)
    mem_w = bs * out_numel

    return {
        'madds': madds,
        'flops': flops,
        'mem_r': numel_to_bytes(mem_r),
        'mem_w': numel_to_bytes(mem_w),
    }

def eval_bn(node):
    params = node['params']
    affine = node['affine']
    running_stats = node['running_stats']
    in_shape = node['in_shape']
    num_feat = node['num_feat']
    bs, in_c, in_h, in_w = in_shape
    in_numel = in_c * in_h * in_w
    
    madds = 4 * in_numel

    flops = bs * in_numel
    if affine: flops *= 2

    mem_r = bs * (in_numel + params)
    mem_w = bs * in_numel

    return {
        'madds': madds,
        'flops': flops,
        'mem_r': numel_to_bytes(mem_r),
        'mem_w': numel_to_bytes(mem_w),
    }

def eval_pool(node):
    params = node['params']
    k = node['kernel_size']
    s = node['stride']
    p = node['padding']
    in_shape = node['in_shape']
    out_shape = node['out_shape']
    kernel_numel = np.prod(k)
    bs = out_shape[0]
    feat = out_shape[1:]
    
    out_numel = np.prod(feat)
    
    madds = (kernel_numel - 1) * out_numel

    flops = kernel_numel * bs * out_numel

    mem_r = bs * (out_numel + params)
    mem_w = bs * out_numel

    return {
        'madds': madds,
        'flops': flops,
        'mem_r': numel_to_bytes(mem_r),
        'mem_w': numel_to_bytes(mem_w),
    }

def eval_act(node):
    params = node['params']
    in_shape = node['in_shape']
    out_shape = node['out_shape']
    bs = out_shape[0]
    num_feat = out_shape[1:]
    out_numel = np.prod(num_feat)

    madds = out_numel
    flops = mem_r = mem_w = bs * out_numel 

    return {
        'madds': madds,
        'flops': flops,
        'mem_r': numel_to_bytes(mem_r),
        'mem_w': numel_to_bytes(mem_w),
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
    out_numel = np.prod(out_feat)

    if mode == "nearest":
        flops = 0
        madds = 0
    if mode == "linear":
        flops = bs * out_numel * 5 # 2 muls + 3 add
        madds = out_numel * 5
    elif mode == "bilinear":
        flops = bs * out_numel * 13 # 6 muls + 7 adds
        madds = out_numel * 13
    elif mode == "bicubic":
        ops_solve_A = 224 # 128 muls + 96 adds
        ops_solve_p = 35 # 16 muls + 12 adds + 4 muls + 3 adds
        ops = (ops_solve_A + ops_solve_p)
        flops = bs * out_numel * ops
        madds = out_numel * ops
    elif mode == "trilinear":
        ops = (13 * 2 + 5)
        flops = bs * out_numel * ops
        madds = out_numel * ops

    mem_r = bs * (out_numel + params)
    mem_w = bs * out_numel

    return {
        'madds': madds,
        'flops': flops,
        'mem_r': numel_to_bytes(mem_r),
        'mem_w': numel_to_bytes(mem_w),
    }

def eval_linear(node):
    in_feat = node['in_feat']
    out_feat = node['out_feat']
    params = node['params']
    bias = node['bias']
    in_shape = node['in_shape']
    out_shape = node['out_shape']
    bs = in_shape[0]
    
    bias_ops = 1 if bias else 0
    madds = out_feat * (in_feat + in_feat - 1 + bias_ops)

    flops = bs * out_feat * in_feat

    mem_r = bs * (in_feat + params)
    mem_w = bs * out_feat

    return {
        'madds': madds,
        'flops': flops,
        'mem_r': numel_to_bytes(mem_r),
        'mem_w': numel_to_bytes(mem_w),
    }

def eval_identity(node):
    return {
        'madds': 0,
        'flops': 0,
        'mem_r': numel_to_bytes(0),
        'mem_w': numel_to_bytes(0),
    }

def eval_nullop(node):
    return {
        'madds': 0,
        'flops': 0,
        'mem_r': numel_to_bytes(0),
        'mem_w': numel_to_bytes(0),
    }


def eval_compute_prop(node):
    if node['updated']: return
    stdtype = node['stdtype']
    if stdtype == 'CONV':
        m_stats = eval_conv(node)
    elif stdtype == 'BN':
        m_stats = eval_bn(node)
    elif stdtype == 'POOL':
        m_stats = eval_pool(node)
    elif stdtype == 'ACT':
        m_stats = eval_act(node)
    elif stdtype == 'UPSMPL':
        m_stats = eval_upsample(node)
    elif stdtype == 'FC':
        m_stats = eval_linear(node)
    elif stdtype == 'IDT':
        m_stats = eval_identity(node)
    else:
        # print(f"compute {stdtype} is not supported!")
        m_stats = eval_nullop(node)
    node.update_values(m_stats)
    node['updated'] = True
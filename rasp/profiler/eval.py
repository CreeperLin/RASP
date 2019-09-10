# -*- coding: utf-8 -*-
import numpy as np

def numel_to_bytes(numel, prec='float'):
    if prec == 'double':
        byte = numel * 8
    elif prec == 'half':
        byte = numel * 2
    elif prec == 'float':
        byte = numel * 4
    else:
        byte = numel * 4
    return int(byte)

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
    in_c = in_shape[1]
    bs = out_shape[0]
    out_c = out_shape[1]
    out_feat = np.prod(out_shape[2:])
    out_numel = out_c * out_feat
    
    kernel_numel = np.prod(kernel_size)
    bias_ops = 1 if bias else 0
    kernel_ops = kernel_numel

    madds = out_numel * (in_c // groups * kernel_ops + bias_ops)
    flops = bs * out_numel * (2 * in_c // groups * kernel_ops - 1 + bias_ops)

    mem_r = bs * (out_numel + params)
    mem_w = bs * out_numel

    return {
        'madds': int(madds),
        'flops': int(flops),
        'mem_r': numel_to_bytes(mem_r),
        'mem_w': numel_to_bytes(mem_w),
    }

def eval_bn(node):
    params = node['params']
    affine = node['affine']
    running_stats = node['running_stats']
    in_shape = node['in_shape']
    num_feat = node['num_feat']
    bs = in_shape[0]
    in_numel = np.prod(in_shape[1:])
    
    affine_ops = 1 if affine else 0
    madds = (affine_ops + 3) * in_numel

    flops = bs * madds

    mem_r = bs * (in_numel + params)
    mem_w = bs * in_numel

    return {
        'madds': int(madds),
        'flops': int(flops),
        'mem_r': numel_to_bytes(mem_r),
        'mem_w': numel_to_bytes(mem_w),
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
    kernel_numel = np.prod(k)

    kernel_mul = 1 if ptype == 'avg' else 0
    kernel_add = kernel_numel - 1
    kernel_ops = kernel_mul + kernel_add

    bs = out_shape[0]
    out_numel = np.prod(out_shape[1:])
    
    madds = kernel_ops * out_numel

    flops = bs * 2 * madds

    mem_r = bs * (out_numel + params)
    mem_w = bs * out_numel

    return {
        'madds': int(madds),
        'flops': int(flops),
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
        'madds': int(madds),
        'flops': int(flops),
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
        madds = out_numel * 5 # 2 muls + 3 add
    elif mode == "bilinear":
        madds = out_numel * 13 # 6 muls + 7 adds
    elif mode == "bicubic":
        ops_solve_A = 224 # 128 muls + 96 adds
        ops_solve_p = 35 # 16 muls + 12 adds + 4 muls + 3 adds
        ops = (ops_solve_A + ops_solve_p)
        madds = out_numel * ops
    elif mode == "trilinear":
        ops = (13 * 2 + 5)
        madds = out_numel * ops
    flops = bs * madds

    mem_r = bs * (out_numel + params)
    mem_w = bs * out_numel

    return {
        'madds': int(madds),
        'flops': int(flops),
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
    madds = out_feat * (in_feat - 1 + bias_ops)

    flops = bs * out_feat * (2 * in_feat - 1 + bias_ops)

    mem_r = bs * (in_feat + params)
    mem_w = bs * out_feat

    return {
        'madds': int(madds),
        'flops': int(flops),
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
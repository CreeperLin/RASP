# -*- coding: utf-8 -*-
import numpy as np
from rasp.profiler.hook import hook_compute, unhook_compute, hook_timing, unhook_timing, hook_all, unhook_all
from rasp.profiler.tree import build_stats_tree, destroy_stats_tree, reset_timing_all
from rasp.utils.config import CFG
from rasp.utils.time import Timer, get_cpu_time
from rasp.utils.reporter import report, save_report, summary
import rasp.frontend as F

def reset_stat(node):
    reset_timing_all(node)

def stat(module, input_shape, disp_fields=None):
    disp_fields = disp_fields or ['name', 'params', 'madds', 'lat', 'net_lat', 'flops', 'mem_r', 'mem_w', 'mem_rw']
    stats_tree = build_stats_tree(module)
    profile_compute_once(module, input_shape)
    profile_timing_once(module, input_shape)
    
    leaves_report = report(list(stats_tree.subnodes()), report_fields=disp_fields, include_root=False)
    # leaves_report = report(list(stats_tree.leaves()), report_fields=disp_fields)
    root_report = report([stats_tree], report_fields=disp_fields, include_root=False)
    print(summary(leaves_report))
    print(summary(root_report))
    
    savepath = CFG.profile.report_savepath
    save_report(leaves_report, savepath, 'leaves_report.csv')
    return stats_tree

def get_batch_data(input_shape):
    batch_size = CFG.profile.batch_size
    data_size = (batch_size, ) + input_shape
    inputs = F.get_random_data(data_size)
    return inputs

def profile_batch(module, inputs):
    num_batches = CFG.profile.num_batches
    warmup_batches = CFG.profile.warmup_batches
    timer = Timer(time_src=get_cpu_time)
    F.pre_run(module, inputs)
    tot_batches = num_batches + warmup_batches
    verbose = CFG.profile.verbose
    for i in range(tot_batches):
        pcnt = round(100 * (i+1) / tot_batches)
        if verbose and pcnt % 25 == 0: print ('RASP prof: {}%'.format(pcnt))
        timer.rec()
        F.run(module, inputs)
        if i < warmup_batches: continue
        timer.rec(include=True)
    # print(timer.stat())
    return timer.mean()

def profile_timing_once(module, input_shape=None, inputs=None):
    inputs = get_batch_data(input_shape) if inputs is None else inputs
    # timing w/o hook
    # lat = profile_batch(module, inputs)
    # timing w/ hook
    stats_tree = profile_timing_on(module)
    lat = profile_batch(module, inputs)
    profile_timing_off(module)
    return stats_tree

def profile_compute_once(module, input_shape=None, inputs=None):
    stats_tree = profile_compute_on(module)
    inputs = get_batch_data(input_shape) if inputs is None else inputs
    F.pre_run(module, inputs)
    F.run(module, inputs)
    profile_compute_off(module)
    return stats_tree

def profile_compute_on(module):
    stats_tree = build_stats_tree(module)
    hook_compute(module)
    reset_stat(stats_tree)
    return stats_tree

def profile_compute_off(module):
    unhook_compute(module)

def profile_timing_on(module):
    stats_tree = build_stats_tree(module)
    hook_timing(module)
    reset_stat(stats_tree)
    return stats_tree

def profile_timing_off(module):
    unhook_timing(module)

def profile_on(module):
    stats_tree = build_stats_tree(module)
    hook_all(module)
    return stats_tree

def profile_off(module):
    unhook_all(module)
    destroy_stats_tree(module)

class profile_timing():
    def __init__(self, module):
        self.module = module
    
    def __enter__(self, module):
        self.stats_tree = profile_timing_on(self.module)
        return self.stats_tree
    
    def __exit__(self):
        profile_off(self.module)

class profile_compute():
    def __init__(self, module):
        self.module = module
    
    def __enter__(self, module):
        self.stats_tree = profile_compute_on(self.module)
        return self.stats_tree
    
    def __exit__(self):
        profile_off(self.module)

class profile_all():
    def __init__(self, module):
        self.module = module
    
    def __enter__(self, module):
        self.stats_tree = profile_on(self.module)
        return self.stats_tree
    
    def __exit__(self):
        profile_off(self.module)
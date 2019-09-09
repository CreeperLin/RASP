# -*- coding: utf-8 -*-
from .eval import eval_compute_prop
import rasp.frontend as F
from rasp.utils.config import CFG
import rasp.device as DEV

class Tape():
    def __init__(self, node, is_set=False):
        self.tape = set() if is_set else list()
        self.add = self.tape.add if is_set else self.tape.append
        self.node = node
        self.accessed = False
    
    def reg_parent(self, name):
        self.accessed = True
        node = self.node
        parent = node.fwd_parent if hasattr(node,'fwd_parent') else node.parent
        while parent is not None:
            tape = getattr(parent, name, None)
            if not tape is None and tape.accessed:
                tape.add(node)
                node.fwd_parent = parent
                break
            parent = parent.parent
    
    @property
    def length(self):
        return len(self.tape)
    
    @property
    def items(self):
        for i in self.tape:
            yield i

    @property
    def items_all(self):
        tape = set() if is_set else list()
        for i in self.tape:
            for it in i.items_all:
                if is_set: tape.add(it)
                else: tape.append(it)
        for i in tape:
            yield i
    
    def clear(self):
        self.tape.clear()


def hook_compute_in(node, in_shape):
    pass

def hook_compute_out(node, in_shape, out_shape):
    tape = node.tape
    node['in_shape'] = in_shape
    node['out_shape'] = out_shape
    if node.num_children==0:
        eval_compute_prop(node)
    else:
        # TODO custom layer measurement
        node['madds'] = 0
        node['flops'] = 0
        node['mem_r'] = 0
        node['mem_w'] = 0
        for n in tape.items:
            node['madds'] += n['madds']
            node['flops'] += n['flops']
            node['mem_r'] += n['mem_r']
            node['mem_w'] += n['mem_w']
    DEV.add_node(node)

def hook_time_start(node, t):
    node['timer'].rec(t=t, include=False)
    node['net_timer'].rec(include=False)

def hook_time_stop(node, t):
    tape = node.tape
    net_timer = node['net_timer']
    net_timer.rec(t=t, include=True)
    net_lat = lat = net_timer.mean()
    for n in tape.items:
        lat -= n['prof_overhead']
        net_lat -= n['tot_lat'] 
    node['lat'] = lat
    node['net_lat'] = net_lat
    timer = node['timer']
    tot_lat = timer.mean() or timer.intv()
    node['tot_lat'] = tot_lat
    overhead = tot_lat - lat
    node['prof_overhead'] = overhead
    timer.rec(include=True)

def hook_all(module):
    hook_compute(module)
    hook_timing(module)

def unhook_all(module):
    unhook_compute(module)
    unhook_timing(module)

def hook_timing(module):
    max_depth = CFG.profile.compute_max_depth
    F.hook_timing(module, max_depth)

def unhook_timing(module):
    F.unhook_timing(module)

def hook_compute(module):
    max_depth = CFG.profile.compute_max_depth
    F.hook_compute(module, max_depth)

def unhook_compute(module):
    F.unhook_compute(module)
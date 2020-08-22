from .eval import eval_compute_prop

class Tape():
    def __init__(self, node):
        self.tape = list()
        self.add = self.tape.append
        self.node = node
        self.accessed = False
    
    def reg_parent(self, name='tape'):
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
        for i in self.tape:
            flag = True
            for it in i.tape.items_all:
                yield it
                flag = False
            if flag: yield i
    
    def clear(self):
        self.tape.clear()


def hook_compute_in(node, in_shape):
    pass

def hook_compute_out(node, in_shape, out_shape, mark_updated=False):
    if node['compute_updated']: return
    tape = node.tape
    node['in_shape'] = in_shape
    node['out_shape'] = out_shape
    if node.num_children==0:
        eval_compute_prop(node)
    elif not tape is None:
        # TODO custom layer measurement
        node['flops'] = 0
        node['mem_r'] = 0
        node['mem_w'] = 0
        node['params'] = 0
        last_n = None
        for n in tape.items:
            if not last_n is None and not n['dev_mem'] is None:
                last_n['dev_mem_delta'] = n['dev_mem'] - last_n['dev_mem']
            node['flops'] += n['flops']
            node['mem_r'] += n['mem_r']
            node['mem_w'] += n['mem_w']
            node['params'] += n['params']
            last_n = n
    if mark_updated:
        node['compute_updated'] = True

def hook_time_start(node, t):
    node['timer'].rec(t=t, include=False)
    node['net_timer'].rec(include=False)

def hook_time_stop(node, t):
    net_timer = node['net_timer']
    net_timer.rec(include=True)
    net_lat = lat = net_timer.mean()
    tape = node.tape
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

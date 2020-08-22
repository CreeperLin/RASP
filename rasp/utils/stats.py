from ..profiler.tree import StatTreeNode, reset_timing_all
from ..profiler.hook import Tape
from ..utils.config import CFG
from ..utils.time import Timer, get_cpu_time
from ..utils.reporter import report, save_report, summary, summary_leaves,\
                             summary_all, summary_tape, summary_node, summary_root
from .. import frontend as F
from .. import device as DEV


def build_stats_tree(module):
    stats_tree = F.get_stats_node(module)
    if stats_tree is None:
        stats_tree = F.reg_stats_node(module)
        # root = StatTreeNode('')
        # root.add_child(stats_tree.name, stats_tree)
        # root.tape = Tape(node=root)
        # root.tape.accessed = True
    # else:
        # root = stats_tree.parent
    return stats_tree


def destroy_stats_tree(module):
    F.unreg_stats_node(module)


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


def reset_stat(node):
    reset_timing_all(node)


def stat(module, input_shape=None, inputs=None, device=None, compute=True, timing=False, memory=False, print_stat=True, returns=None,
         report_type='tape', include_root=False, report_fields=None, includes=None, excludes=None, save_path=None, keep_tree=False):
    excludes = [] if excludes is None else excludes
    includes = [] if includes is None else includes
    DEV.set_device(device)
    destroy_stats_tree(module)
    stats_tree = build_stats_tree(module)
    if memory:
        stat_memory_on()
        includes.extend(['net_dev_mem', 'dev_mem_alloc', 'dev_max_mem_alloc'])
    if compute:
        profile_compute_once(module, input_shape, inputs, device)
    if timing:
        profile_timing_once(module, input_shape, inputs, device)
    else:
        excludes.extend(['lat', 'net_lat', 'lat[%]', 'FLOPS'])
    if report_type is None:
        reporter = None
    elif report_type == 'all':
        reporter = summary_all
    elif report_type == 'tape':
        reporter = summary_tape
    elif report_type == 'node':
        reporter = summary_node
    elif report_type == 'leaves':
        reporter = summary_leaves
    elif report_type == 'root':
        reporter = summary_root
    else:
        raise ValueError('unsupported report type: {}'.format(report_type))
    df = None
    if not reporter is None:
        sum_str, df = reporter(stats_tree, include_root, report_fields, includes, excludes)
        if not save_path is None:
            save_report(df, save_path)
        if print_stat: 
            print(sum_str)
    if not keep_tree:
        destroy_stats_tree(module)
    if not returns: return
    ret_types = returns.split(',')
    rets = []
    for ret_type in ret_types:
        ret_type = ret_type.strip()
        if ret_type == 'data':
            ret = df
        elif ret_type == 'tree':
            ret = stats_tree
        elif ret_type == 'sum':
            ret = sum_str
        else:
            raise ValueError('invalid return type: {}'.format(ret_type))
        rets.append(ret)
    return rets[0] if len(rets) == 1 else rets


def get_batch_data(input_shape, inputs):
    if not inputs is None:
        if not input_shape is None:
            raise ValueError("input_shape is not none")
        return inputs
    if input_shape is None:
        raise ValueError("input_shape required")
    if len(input_shape) <= 3:
        batch_size = CFG.profile.batch_size
        data_size = (batch_size, ) + input_shape
    else:
        data_size = input_shape
    inputs = F.get_random_data(data_size)
    return inputs


def profile_batch(module, inputs, device):
    num_batches = CFG.profile.num_batches
    warmup_batches = CFG.profile.warmup_batches
    timer = Timer(time_src=get_cpu_time, synch=DEV.get_synchronize())
    tot_batches = num_batches + warmup_batches
    verbose = CFG.profile.verbose
    with F.get_ctx(module, inputs, device) as ctx:
        for i in range(tot_batches):
            pcnt = round(100 * (i+1) / tot_batches)
            if verbose and pcnt % 25 == 0: print ('RASP prof: {}%'.format(pcnt))
            timer.rec()
            ctx.run()
            if i < warmup_batches: continue
            timer.rec(include=True)
    # print(timer.stat())
    return timer.mean()


def stat_memory_on():
    CFG.frontend.stat_mem = True
    

def stat_memory_off():
    CFG.frontend.stat_mem = False


def profile_compute_once(module, input_shape=None, inputs=None, device=None):
    stats_tree = profile_compute_on(module)
    inputs = get_batch_data(input_shape, inputs)
    F.run(module, inputs, device)
    profile_compute_off(module)
    return stats_tree


def profile_compute(module):
    inputs = get_batch_data(input_shape, inputs)
    F.run(module, inputs, device)


def profile_compute_on(module):
    stats_tree = build_stats_tree(module)
    hook_compute(module)
    reset_stat(stats_tree)
    return stats_tree


def profile_compute_off(module):
    unhook_compute(module)


def profile_timing_once(module, input_shape=None, inputs=None, device=None):
    inputs = get_batch_data(input_shape, inputs)
    # timing w/o hook
    # lat = profile_batch(module, inputs)
    # timing w/ hook
    stats_tree = profile_timing_on(module)
    lat = profile_batch(module, inputs, device)
    profile_timing_off(module)
    return stats_tree


def profile_timing(module, input_shape, inputs, device):
    inputs = get_batch_data(input_shape, inputs)
    lat = profile_batch(module, inputs, device)
    return lat


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


class prof_timing():
    def __init__(self, module):
        self.module = module
    
    def __enter__(self, module):
        self.stats_tree = profile_timing_on(self.module)
        return self.stats_tree
    
    def __exit__(self):
        profile_off(self.module)


class prof_compute():
    def __init__(self, module):
        self.module = module
    
    def __enter__(self, module):
        self.stats_tree = profile_compute_on(self.module)
        return self.stats_tree
    
    def __exit__(self):
        profile_off(self.module)

class prof_all():
    def __init__(self, module):
        self.module = module
    
    def __enter__(self, module):
        self.stats_tree = profile_on(self.module)
        return self.stats_tree
    
    def __exit__(self):
        profile_off(self.module)

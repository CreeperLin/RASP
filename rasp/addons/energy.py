from ..profiler.eval import add_evaluator
from ..utils.reporter import add_unit

energy_cost_table = {
    'int16_add': 0.06,
    'int16_mult': 0.8,
    'int32_add': 0.1,
    'int32_mult': 3.1,
    'int64_add': 0.2,       # spec
    'int64_mult': 12.5,     # spec
    'float16_add': 0.45,
    'float16_mult': 1.1,
    'float32_add': 0.9,
    'float32_mult': 4.5,
    'float64_add': 1.8,     # spec
    'float64_mult': 18.0,   # spec
    'int32_reg': 1.0,
    'int32_l1cache': 3.5,
    'int32_l2cache': 30.2,
    'int32_chipdram': 160.0,
    'int32_dram': 640.0,
    'float32_reg': 1.0,
    'float32_l1cache': 3.5,
    'float32_l2cache': 30.2,
    'float32_chipdram': 160.0,
    'float32_sram': 5.0,
    'float32_dram': 640.0,
}


def get_energy_cost(dtype, key):
    return energy_cost_table.get('{}_{}'.format(dtype, key), None)


def set_energy_cost(dtype, key, val):
    energy_cost_table['{}_{}'.format(dtype, key)] = val


def eval_energy_default(node):
    dtype = node['dtype']
    if dtype is None:
        dtype = 'float32'
    energy = node['flops'] * get_energy_cost(dtype, 'mult') + (node['mem_r'] + node['mem_w']) * get_energy_cost(dtype, 'dram')
    return {
        'energy': energy
    }


add_evaluator('all', eval_energy_default)
add_unit('energy', 'pJ')

# -*- coding: utf-8 -*-
import os
import pandas as pd

pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 10000)
pd.set_option('display.max_columns', 10000)

def round_value(value, binary=False):
    divisor = 1024. if binary else 1000.

    if value // divisor**4 > 0:
        return str(round(value / divisor**4, 2)) + 'T'
    elif value // divisor**3 > 0:
        return str(round(value / divisor**3, 2)) + 'G'
    elif value // divisor**2 > 0:
        return str(round(value / divisor**2, 2)) + 'M'
    elif value // divisor > 0:
        return str(round(value / divisor, 2)) + 'K'
    return str(value)


def report(collected_nodes, include_root=False, report_fields=None):
    if len(collected_nodes)==0: return None
    all_fields = ['name', 'type', 'in_shape', 'out_shape',
            'params', 'madds', 'lat', 'net_lat', 'lat[%]', 'prof_overhead', 'flops', 'mem_r', 'mem_w', 'mem_rw']
    if report_fields is None: report_fields = all_fields
    if include_root:
        root = collected_nodes[0].root
        collected_nodes.append(root)
    data = list()
    for node in collected_nodes:
        series = []
        for f in report_fields:
            if f == 'mem_rw':
                val = (node['mem_r'] or 0.0) + (node['mem_w'] or 0.0)
            elif node[f] is None:
                val = '' if f == 'name' else 0
            elif f == 'in_shape' or f == 'out_shape':
                val = ' '.join(['{:>3d}'] * len(node[f])).format(*[e for e in node[f]])
            else:
                val = node[f]
            series.append(val)
        data.append(series)
    df = pd.DataFrame(data)
    df.columns = report_fields

    # df = df.fillna(0)
    df = df.fillna(' ')

    df.drop([f for f in df.columns if not f in report_fields],axis=1,inplace=True)
    return df

def summary(df):
    total_params = df['params'].sum()
    total_madds = df['madds'].sum()
    total_flops = df['flops'].sum()
    total_mread = df['mem_r'].sum()
    total_mwrite = df['mem_w'].sum()
    total_memrw = df['mem_rw'].sum()
    df['lat[%]'] = 100 * df['net_lat'] / (df['net_lat'].sum() + 1e-16)
    total_latency_ratio = df['lat[%]'].sum()
    total_latency = df['lat'].sum()
    total_net_latency = df['net_lat'].sum()
    # del df['lat']
    # Add Total row
    total_df = pd.Series(['', total_params,
                    total_madds, total_flops,
                    total_latency_ratio, total_latency, total_net_latency, total_mread, total_mwrite, total_memrw],
                    index=['name', 'params', 'madds', 'flops', 'lat[%]', 'lat', 'net_lat', 'mem_r', 'mem_w', 'mem_rw'],
                    name='total')
    df = df.append(total_df)

    sum_str = str(df) + '\n'
    sum_str += "=" * len(str(df).split('\n')[0])
    sum_str += '\n'
    sum_str += "Total params: {:,}\n".format(total_params)
    sum_str += "-" * len(str(df).split('\n')[0])
    sum_str += '\n'
    sum_str += "Total MAdd: {} MAdds\n".format(round_value(total_madds))
    sum_str += "Total FLOPs: {} FLOPs\n".format(round_value(total_flops))
    sum_str += "Total Mem(R+W): {}B\n".format(round_value(total_memrw, True))
    sum_str += "Total Latency: {:.5f} ms\n".format(total_latency)
    # sum_str += "Total Profile Overhead: {:.5f} ms\n".format(total_prof_overhead)
    return sum_str, total_df

def summary_leaves(node, include_root=False, report_fields=None):
    return summary(report(list(node.leaves()), include_root=include_root, report_fields=report_fields))

def summary_all(node, include_root=False, report_fields=None):
    return summary(report(list(node.subnodes()), include_root=include_root, report_fields=report_fields))

def summary_node(node, include_root=False, report_fields=None):
    return summary(report([node], include_root=include_root, report_fields=report_fields))

def load_report(path):
    df = pd.read_csv(path, converters={'name':str})
    return df

def save_report(report, savepath, name):
    out_dir = os.path.join('.', savepath)
    os.makedirs(out_dir, exist_ok=True)
    report.to_csv(os.path.join(out_dir, name), index=False)

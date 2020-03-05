# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np

pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 10000)
pd.set_option('display.max_columns', 10000)

def get_unit_name(field_name):
    if 'mem' in field_name:
        return 'Byte'
    if 'lat' in field_name:
        return 'MS'
    return ''

def round_value(value, binary=False, prec=2):
    divisor = 1024. if binary else 1000.

    if value // divisor**4 > 0:
        return str(round(value / divisor**4, prec)) + 'T'
    elif value // divisor**3 > 0:
        return str(round(value / divisor**3, prec)) + 'G'
    elif value // divisor**2 > 0:
        return str(round(value / divisor**2, prec)) + 'M'
    elif value // divisor > 0:
        return str(round(value / divisor, prec)) + 'K'
    return str(round(value, prec))


def report(collected_nodes, include_root=False, report_fields=None):
    if len(collected_nodes)==0: return None
    all_fields = ['name', 'type', 'in_shape', 'out_shape', 'params',
            'lat', 'net_lat', 'lat[%]', 'flops', 'FLOPS', 'mem_r', 'mem_w', 'mem_rw',
            'dev_mem_alloc', 'dev_mem_delta']
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
                val = str(node[f])
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
    if df is None: return '', None
    df_fields = df.columns
    rep_fields = list(df_fields)
    if 'lat[%]' in df_fields and 'net_lat' in df_fields:
        df['lat[%]'] = 100 * df['net_lat'] / (df['net_lat'].sum() + 1e-16)
    if 'FLOPS' in df_fields and 'net_lat' in df_fields and 'flops' in df_fields:
        df['FLOPS'] = 1000 * df['flops'] / df['net_lat']
    agg_fields = ['name']
    agg_val = ['']
    for f in rep_fields:
        try:
            if f == 'FLOPS':
                tot_val = 1000 * df['flops'].sum() / df['net_lat'].sum()
            elif f in ['name', 'type', 'in_shape', 'out_shape']: 
                continue
            else:
                tot_val = df[f].sum()
            agg_val.append(tot_val)
            agg_fields.append(f)
        except:
            continue

    total_df = pd.Series(agg_val, index=agg_fields, name='total')
    df = df.append(total_df)

    sum_str = str(df) + '\n'
    sum_str += "=" * len(str(df).split('\n')[0])
    sum_str += '\n'
    for f, v in zip(agg_fields, agg_val):
        if f in ['name', 'type', 'in_shape', 'out_shape', 'lat[%]']: continue
        binary = False
        if f in ['mem_r', 'mem_w', 'mem_rw', 'dev_mem_alloc', 'dev_mem_delta']:
            binary = True
        sum_str += "Total {}: {} {}\n".format(f, round_value(v, binary, prec=2), get_unit_name(f))
    return sum_str, total_df

def summary_leaves(node, include_root=False, report_fields=None):
    return summary(report(list(node.leaves()), include_root=include_root, report_fields=report_fields))

def summary_all(node, include_root=False, report_fields=None):
    return summary(report(list(node.subnodes()), include_root=include_root, report_fields=report_fields))

def summary_tape(node, include_root=False, report_fields=None):
    return summary(report(list(node.tape.items_all), include_root=include_root, report_fields=report_fields))

def summary_node(node, include_root=False, report_fields=None):
    return summary(report([node], include_root=include_root, report_fields=report_fields))

def summary_root(node, include_root=False, report_fields=None):
    return summary(report([node.root], include_root=include_root, report_fields=report_fields))

def load_report(path):
    df = pd.read_csv(path, converters={'name':str})
    return df

def save_report(report, savepath):
    save_dir = os.path.split(savepath)[0]
    os.makedirs(save_dir, exist_ok=True)
    report.to_csv(savepath, index=False)

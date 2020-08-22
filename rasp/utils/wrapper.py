import sys
import time
import numpy as np
from functools import wraps
from .. import frontend as F

t0 = 0
m0 = 0
mtable = {}
ttable = {}


def get_cputime():
    return time.perf_counter()


def seqstat(arr):
    a = np.array(arr)
    return '[ {:.3f} / {:.3f} / {:.3f} / {:.3f} ]'.format(
        np.mean(a), np.min(a), np.max(a), np.std(a))


def report_time(msg=''):
    global t0
    t1 = get_cputime()
    fr = sys._getframe(1)
    print("CPU Time: {} {} @ {} : {:.3f} dt: {:.3f} sec".format(
        msg.center(20, ' '), fr.f_code.co_name, fr.f_lineno, t1, t1 - t0))
    t0 = t1


def report_mem(msg=''):
    global m0
    m1 = F.get_current_mem()
    fr = sys._getframe(1)
    print("GPU Mem: {} {} @ {} : {:.3f} dt: {:.3f} MB".format(
        msg.center(20, ' '), fr.f_code.co_name, fr.f_lineno, m1, m1 - m0))
    m0 = m1


def profile_mem(function):
    @wraps(function)
    def gpu_mem_profiler(*args, **kwargs):
        m1 = F.get_current_mem()
        result = function(*args, **kwargs)
        m2 = F.get_current_mem()
        fp = m2 - m1
        fname = function.__name__
        if fname in mtable:
            mtable[fname].append(fp)
        else:
            mtable[fname] = [fp]
        print("GPU Mem: {}: {:.3f} / {:.3f} / {:.3f} / {} MB".format(
            fname.center(20, ' '), m1, m2, fp, seqstat(mtable[fname])))
        return result

    return gpu_mem_profiler


def profile_time(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t1 = get_cputime()
        result = function(*args, **kwargs)
        t2 = get_cputime()
        lat = t2 - t1
        fname = function.__name__
        if fname in ttable:
            ttable[fname].append(lat)
        else:
            ttable[fname] = [lat]
        print("CPU Time: {}: {:.3f} / {:.3f} / {:.3f} / {} sec".format(
            fname.center(20, ' '), t1, t2, lat, seqstat(ttable[fname])))
        return result

    return function_timer


class profile_ctx():
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.m0 = F.get_current_mem()
        self.t0 = get_cputime()

    def __exit__(self):
        self.t1 = get_cputime()
        self.m1 = F.get_current_mem()
        lat = self.t1 - self.t0
        mem = self.m1 - self.m0
        fname = self.name
        if fname in ttable:
            ttable[fname].append(lat)
        else:
            ttable[fname] = [lat]
        if fname in mtable:
            mtable[fname].append(mem)
        else:
            mtable[fname] = [mem]
        print("CPU Time: {}: {:.3f} / {:.3f} / {:.3f} / {} sec".format(
            fname.center(20, ' '), self.t0, self.t1, lat,
            seqstat(ttable[fname])))
        print("GPU Mem: {}: {:.3f} / {:.3f} / {:.3f} / {} MB".format(
            fname.center(20, ' '), self.m0, self.m1, mem,
            seqstat(mtable[fname])))

    def report(self):
        t1 = get_cputime()
        fr = sys._getframe(1)
        print("CPU Time: {} {} @ {} : {:.3f} dt: {:.3f} sec".format(
            self.name.center(20, ' '), fr.f_code.co_name, fr.f_lineno, t1,
            t1 - self.t0))
        m1 = F.get_current_mem()
        print("GPU Mem: {} {} @ {} : {:.3f} dt: {:.3f} MB".format(
            self.name.center(20, ' '), fr.f_code.co_name, fr.f_lineno, m1,
            m1 - self.m0))

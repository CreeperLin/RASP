# -*- coding: utf-8 -*-
import time
import numpy as np

def get_cpu_time():
    return time.perf_counter() * 1000

def get_time():
    return get_cpu_time()

class Timer():
    def __init__(self, time_src, synch=None, store=True):
        self.time_src = time_src or get_cpu_time
        self.record = np.array([])
        self.synch = synch or (lambda: None)
        self.store = store
        self.last_t = self.time_src()
    
    def rec(self, t=None, include=False):
        self.synch()
        if t is None: t = self.time_src()
        if include:
            dt = (t - self.last_t) 
            if self.store: self.record = np.append(self.record, dt)
            else: self.record = np.array([dt])
        self.last_t = self.time_src()
    
    def intv(self, t=None):
        self.synch()
        if t is None: t = self.time_src()
        return (t - self.last_t)

    def get_record(self):
        return self.record
    
    def stat(self):
        rec = self.record
        return None if len(rec)==0 else (len(rec), np.sum(rec), np.mean(rec), np.min(rec), np.max(rec), np.std(rec))

    def reset(self):
        self.record = np.array([])
        self.last_t = self.time_src()
    
    def mean(self):
        return None if len(self.record)==0 else np.mean(self.record)

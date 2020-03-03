# -*- coding: utf-8 -*-
from ..utils.reporter import report

class RegressorDevice():
    def __init__(self, regressor):
        self.tape = []
        self.reg = regressor

    def reset(self):
        self.tape = []

    def add_node(self, node):
        self.tape.append(node)

    def run_node(self, node):
        pass

    def run(self, df=None, X_col=None, y_col=None, X_filter=None, y_filter=None):
        X_col = self.reg.var_observe if X_col is None else X_col
        y_col = self.reg.var_target if y_col is None else y_col
        fields = list(set(X_col+y_col))
        df = report(self.tape, include_root=False, report_fields=fields) if df is None else df

        X = self.reg.filter(df, X_col, row_filter=X_filter)
        y = self.reg.filter(df, y_col, row_filter=y_filter)

        X = X.reshape(y.shape[0],-1)
    
        self.reg.test(X, y)
        return self.reg.predict(X)

device = RegressorDevice()

def init(*args, **kwargs):
    global device
    device = RegressorDevice(*args, **kwargs)

def reset():
    device.reset()

def add_node(node):
    device.add_node(node)

def get_tape():
    return device.tape

def get_data(X_col, y_col, fields):
    X_col = self.reg.var_observe if X_col is None else X_col
    y_col = self.reg.var_target if y_col is None else y_col
    fields = list(set(X_col+y_col)) if fields is None else fields
    return report(device.tape, include_root=False, report_fields=fields)

def run(*args, **kwargs):
    return device.run(*args, **kwargs)
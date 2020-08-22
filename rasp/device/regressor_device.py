from ..utils.reporter import report


class RegressorDevice():
    tape = []
    regressor = None

    @staticmethod
    def init(regressor):
        RegressorDevice.tape = []
        RegressorDevice.regressor = regressor

    @staticmethod
    def reset():
        RegressorDevice.tape = []

    @staticmethod
    def add_node(node):
        RegressorDevice.tape.append(node)

    @staticmethod
    def run_node(node):
        pass

    @staticmethod
    def run(df=None, X_col=None, y_col=None, X_filter=None, y_filter=None):
        X_col = RegressorDevice.regressor.var_observe if X_col is None else X_col
        y_col = RegressorDevice.regressor.var_target if y_col is None else y_col
        fields = list(set(X_col + y_col))
        df = report(RegressorDevice.tape,
                    include_root=False,
                    report_fields=fields) if df is None else df

        X = RegressorDevice.regressor.filter(df, X_col, row_filter=X_filter)
        y = RegressorDevice.regressor.filter(df, y_col, row_filter=y_filter)

        X = X.reshape(y.shape[0], -1)

        RegressorDevice.regressor.test(X, y)
        return RegressorDevice.regressor.predict(X)


init = RegressorDevice.init

reset = RegressorDevice.reset

add_node = RegressorDevice.add_node

run = RegressorDevice.run


def get_tape():
    return RegressorDevice.tape


def get_data(X_col, y_col, fields):
    X_col = RegressorDevice.regressor.var_observe if X_col is None else X_col
    y_col = RegressorDevice.regressor.var_target if y_col is None else y_col
    fields = list(set(X_col + y_col)) if fields is None else fields
    return report(RegressorDevice.tape,
                  include_root=False,
                  report_fields=fields)

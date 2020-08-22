import os
import sys
import numpy as np
import sklearn
from ..utils.config import CFG

def get_linear_model():
    from sklearn.linear_model import LinearRegression
    model = LinearRegression(

    )
    return model

def get_ridge_model():
    from sklearn.linear_model import Ridge
    model = Ridge(
        alpha=.5,
    )
    return model

def get_xgboost_model():
    import xgboost as xgb
    model = xgb.XGBRegressor(
                learning_rate =0.1,
                n_estimators=1000,
                max_depth=5,
                gamma=0,
                subsample=0.8,
                colsample_bytree=0.8,
                nthread=4,
                seed=27)
    return model

def get_mlp_model():
    from sklearn.neural_network import MLPRegressor
    model = MLPRegressor(
        hidden_layer_sizes=(3, ),
        alpha=.5,
        verbose = True,
    )
    return model

model_creator = {
    'linear': get_linear_model,
    'ridge': get_ridge_model,
    'xgb': get_xgboost_model,
    'mlp': get_mlp_model,
}

def get_regress_model(model_desc):
    if model_desc in model_creator:
        model = model_creator[model_desc]()
    else:
        raise ValueError('unsupported model: {}'.format(model_desc))
    return model

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

class Regressor():
    def __init__(self, model=None):
        super().__init__()
        model = model or CFG.analysis.regress_model
        self.model = get_regress_model(model)

    def set_vars(self, observe, target):
        self.var_observe = observe or self.var_observe
        self.var_target = target or self.var_target
        print('regress: vars: obs:{} tgt:{}'.format(self.var_observe, self.var_target))
    
    def filter(self, df, col_name, row_filter=None):
        df = df if row_filter is None else df[row_filter]
        data = []
        for n in col_name:
            item = df[n] if len(df)==1 else df[n][:1].item()
            if isinstance(item, str):
                enc = OneHotEncoder(sparse=False)
                d = enc.fit_transform(df[n].to_numpy().reshape(-1, 1))
            elif isinstance(item, tuple):
                d = np.array([i[0] for i in df[n]]).reshape(-1,1)
            else:
                d = df[n].to_numpy().reshape(-1,1)
            data.append(d)
        data = np.concatenate(data, axis=-1)

        print('regress: filter: {}'.format(data.shape))
        return data

    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)

    def test(self, X, y):
        y_pred = self.model.predict(X)
        print('regress r2: {}'.format(r2_score(y, y_pred)))
        print('regress rmse: {}'.format(mean_squared_error(y, y_pred) ** 0.5))
        print('regress mae: {}'.format(mean_absolute_error(y, y_pred)))
    
    def train(self, X, y):
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        print(x_train.shape)
        print(y_train.shape)
        print(x_test.shape)
        print(y_test.shape)
        self.fit(x_train, y_train)
        self.test(x_test, y_test)

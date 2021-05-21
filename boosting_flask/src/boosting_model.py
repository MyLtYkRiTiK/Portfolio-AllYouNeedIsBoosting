import copy

import daal4py as d4p
import lightgbm as lgb
import numpy as np
from datatable import dt, unique
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

from .metrics import f1_binary


class Trainer:
    def __init__(self, dataframe_train: dt.Frame,
                 dataframe_test: [dt.Frame, None],
                 train_columns: list,
                 user_target: str,
                 metrics: list,
                 user_type: str,
                 user_balance: str):
        self.dataframe_train = dataframe_train
        self.dataframe_test = dataframe_test
        self.train_columns = train_columns
        self.target = user_target
        self.metrics = metrics
        self.task_type = user_type
        self.unbalanced = True if user_balance == 'unbalanced' else False
        self.params = {'task': 'train',
                       'max_depth': 8,
                       'min_child_weight': 25,
                       'bagging_fraction': 0.7,
                       'bagging_freq': 10,
                       'feature_fraction': 0.9,
                       'lambda_l2': 2,
                       'objective': self.task_type,
                       'is_unbalance': self.unbalanced,
                       'data_random_seed': 42,
                       'verbose': -1,
                       'num_leaves': 32,
                       }
        self.feval = []
        self.best_iteration = []
        self.best_score = []
        self.n_splits = 5
        self.update_parameters( )
        self.update_metrics( )

    def update_parameters(self):
        if self.task_type == 'multiclassova':
            unique_target_values = set(unique(self.dataframe_train[:, self.target]).to_dict( )[self.target])
            self.params['num_class'] = len(unique_target_values)

    def update_metrics(self):
        if self.task_type == 'regression':
            self.params['metric'] = self.metrics
        else:
            self.params['metric'] = 'None'
            if 'logloss' in self.metrics:
                self.params['metric'] = self.task_type
            if 'f1' in self.metrics:
                if self.task_type == 'binary':
                    self.feval.append(f1_binary)
                elif self.task_type == 'multiclassova':
                    def f1_multiclass(y_hat, train_data):
                        y_true = train_data.get_label( )
                        y_hat = y_hat.reshape(self.params['num_class'], -1).T
                        y_hat = np.argmax(y_hat, axis=1)
                        return 'f1', f1_score(y_true, y_hat, average='macro'), True

                    self.feval.append(f1_multiclass)

    def train_cv(self, cv_train, cv_train_target, cv_valid, cv_valid_target):
        lgb_train = lgb.Dataset(cv_train,
                                cv_train_target,
                                feature_name=self.train_columns)
        lgb_valid = lgb.Dataset(cv_valid,
                                cv_valid_target,
                                feature_name=self.train_columns)

        lgb_model = lgb.train(params=self.params,
                              train_set=lgb_train,
                              num_boost_round=1500,
                              valid_sets=[lgb_valid],
                              verbose_eval=-1,
                              learning_rates=lambda iterate: 0.025 * (0.99 ** iterate),
                              early_stopping_rounds=150,
                              feval=self.feval)
        self.best_iteration.append(lgb_model.best_iteration)
        self.best_score.append(list(lgb_model.best_score['valid_0'].items( )))

    def train_final(self):
        train = copy.deepcopy(self.dataframe_train[:, self.train_columns])
        train_target = copy.deepcopy(self.dataframe_train[:,
                                     self.target].to_dict( )[self.target])
        lgb_train = lgb.Dataset(train,
                                train_target,
                                feature_name=self.train_columns)

        lgb_model = lgb.train(self.params,
                              lgb_train,
                              num_boost_round=int(np.median(self.best_iteration)),
                              verbose_eval=False,
                              learning_rates=lambda iterate: 0.025 * (0.99 ** iterate),
                              feval=self.feval,
                              )

        daal_model = d4p.get_gbt_model_from_lightgbm(lgb_model)
        return daal_model

    def calculate_cv_scores(self):
        cv_scores = {x[0]: 0 for x in self.best_score[0]}
        for metrics in self.best_score:
            for score in metrics:
                cv_scores[score[0]] += score[1]
        metric_mapper = {'l2': 'mse',
                         'l1': 'mae',
                         'rmse': 'rmse',
                         'f1': 'f1',
                         'binary_logloss': 'logloss',
                         'multi_logloss': 'logloss'}
        return {metric_mapper[k]: f'{v / self.n_splits:.3f}' for k, v in cv_scores.items( )}

    def train(self):
        kfold = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        for train_ids, valid_ids in kfold.split(list(range(self.dataframe_train.shape[0]))):
            cv_train = copy.deepcopy(self.dataframe_train[train_ids, self.train_columns]).to_numpy( )
            cv_train_target = copy.deepcopy(self.dataframe_train[train_ids,
                                                                 self.target].to_dict( )[self.target])
            cv_valid = copy.deepcopy(self.dataframe_train[valid_ids, self.train_columns]).to_numpy( )
            cv_valid_target = copy.deepcopy(self.dataframe_train[valid_ids,
                                                                 self.target].to_dict( )[self.target])
            self.train_cv(cv_train, cv_train_target, cv_valid, cv_valid_target)

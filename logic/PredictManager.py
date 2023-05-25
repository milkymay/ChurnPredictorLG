from scipy.stats import uniform, randint, loguniform

from predictors.ChurnPredictor import ChurnPredictor
from predictors.base.LogisticPredictor import LogisticPredictor
from predictors.base.NoPredictor import NoPredictor
from predictors.base.RandomPredictor import RandomPredictor
from utils.Data import Data
from utils.PrettyLogger import PrettyLogger

model_params = {
    'forest': {
        'model__max_depth': randint(4, 13),
        'model__max_features': [None, 'sqrt', 'log2', 0.2, 0.5, 0.8, 10, 20],
        'model__n_estimators': randint(300, 501),
        'model__criterion': ['gini', 'entropy'],
    },
    'lightgbm': {
        'model__learning_rate': [0.1, 0.05, 0.01, 0.005],
        'model__num_leaves': randint(5, 128),
        'model__max_depth': [-1, 5, 10, 15, 20, 25],
        'model__min_child_samples': randint(5, 30),
        'model__min_child_weight': uniform(loc=0, scale=100),
        'model__colsample_bytree': uniform(loc=0.6, scale=0.3),
        'model__subsample': uniform(loc=0.6, scale=0.3),
        'model__subsample_freq': randint(0, 400),
        'model__reg_alpha': uniform(loc=0, scale=10),
        'model__reg_lambda': uniform(loc=0, scale=10),
        'model__max_bin': [63, 127, 255, 511, 1023, 2047],
        'model__min_split_gain': uniform(loc=0, scale=0.5)
    },
    'catboost': {
        'model__learning_rate': loguniform(1e-4, 1e-1),
        'model__depth': randint(2, 12),
        'model__l2_leaf_reg': randint(1, 10),
        'model__bagging_temperature': uniform(0, 2),
        'model__border_count': randint(32, 254)
    },
    'nn': {
        'model__epochs': [100, 200, 300],
        'model__batch_size': randint(16, 64),
        # 'model__optimizer__learning_rate': [0.001, 0.01, 0.1]
    }
    # 'forest': {
    #     'model__max_depth': [4, 6, 8, 10, 12],
    #     'model__max_features': ['sqrt', 'log2', None, 0.5, 0.7, 0.9],
    #     'model__n_estimators': [300, 350, 400, 450, 500],
    #     'model__criterion': ['gini', 'entropy'],
    #     # 'model__min_samples_split': [2, 5],
    #     # 'model__min_samples_leaf': [1, 2, 4]
    # },
    # 'lightgbm': {
    #     'model__task': ['train'],
    #     'model__boosting_type': ['gbdt'],
    #     'model__learning_rate': [0.1, 0.05, 0.01, 0.005],
    #     'model__num_leaves': [7, 15, 31, 63, 127],
    #     'model__max_depth': [-1, 5, 10, 15, 20, 25],
    #     'model__min_data_in_leaf': [5, 10, 15, 20, 25, 30],
    #     'model__min_sum_hessian_in_leaf': [0.001, 0.01, 0.1, 1, 10, 100],
    #     'model__feature_fraction': [0.6, 0.7, 0.8, 0.9],
    #     'model__bagging_fraction': [0.6, 0.7, 0.8, 0.9],
    #     'model__bagging_freq': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 100, 200, 400],
    #     'model__lambda_l1': [0, 0.1, 0.5, 1, 2, 5, 10],
    #     'model__lambda_l2': [0, 0.1, 0.5, 1, 2, 5, 10],
    #     'model__max_bin': [63, 127, 255, 511, 1023, 2047],
    #     'model__min_gain_to_split': [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    # }
    # 'catboost': {
    #     'model__learning_rate': [0.0001, 0.001, 0.01, 0.03, 0.1],
    #     'model__depth': [2, 4, 6, 8, 10, 12],
    #     'model__l2_leaf_reg': [1, 3, 5, 7, 9],
    #     'model__bagging_temperature': [0, 0.1, 0.2, 0.5, 1, 2],
    #     'model__border_count': [32, 64, 96, 128, 192, 254]
    # }
}


def model_to_metrics(model, clfs, data, metrics, simple=False):
    if simple:
        model.fit(data.X_train, data.y_train, data.X_test, data.y_test)
    else:
        model.fit(data.X_train, data.y_train)
    metrics[model.model_name] = model.get_metrics(data.X_test, data.y_test)
    clfs[model.model_name] = model


class PredictManager:
    def __init__(self, data_file_path, log_file_path=None, clfs_names=None, text_analysis=None):
        if clfs_names is None:
            clfs_names = ['forest', 'catboost', 'lightgbm']
        if text_analysis is None:
            text_analysis = []
        self.metrics = {}
        self.clfs_names = clfs_names
        self.clfs = {}
        self.text_analysis = text_analysis
        self.data = Data(data_file_path, False, text_analysis)
        self.logger = PrettyLogger(log_file_path)

    def baseline(self):
        model_to_metrics(RandomPredictor(), self.clfs, self.data, self.metrics)
        self.log_metrics("random")
        model_to_metrics(LogisticPredictor(), self.clfs, self.data, self.metrics, True)
        self.log_metrics("logistic")
        self.metrics["no_zero"] = NoPredictor().get_metrics(self.data.X_test, self.data.y_test)
        self.log_metrics("no_zero")

    def log_metrics(self, type, best_params=None, feature_importances=None):
        self.logger.log_results(self.text_analysis, type, self.metrics[type], best_params=best_params,
                                feature_importances=feature_importances)

    def evaluate(self):
        for clf_name in self.clfs_names:
            model = ChurnPredictor(model_name=clf_name, hyperparams=model_params[clf_name])
            model_to_metrics(model, self.clfs, self.data, self.metrics)
            self.log_metrics(clf_name, best_params=model.results['best_params'],
                             feature_importances=model.results['feature_importances'])

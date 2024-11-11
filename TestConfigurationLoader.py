# Author: Marcelo Costalonga

import os
from enum import Enum
from dotenv import load_dotenv

DATASET_TYPES = {
    'fake': 0,
    'original': 1
}
FS_SCORE_METHODS = {
    'affinity': 0,
    'chi2': 1,
    'anovaf': 2,
    'pearson': 3
}
CROSS_VALS = {
    'kfold': 0,
    'series': 1,
    'none': 2,
    'grid_search': 2
}


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class TestConfiguration(metaclass=Singleton):
    env_file_path = ''

    def __init__(self, file_path=None):
        self.used_dataset = None
        self.used_fs = None
        self.used_score_method = None
        self.used_cross_val_method = None
        self.tf_idf_config = None
        self.number_of_splits = None
        self.seed = None
        self.env_spec = []
        self.env_file_path = file_path
        self.gs_cv = None
        self.gs_scoring = None

        # Loading values from env file
        self.load_env_vars()

    def load_env_vars(self):
        if load_dotenv(self.env_file_path):
            self.used_dataset = os.getenv('DATASET', 'original').lower()
            self.used_fs = os.getenv('USED_FEATURE_SELECTION', 'false').lower() == 'true'
            self.used_score_method = os.getenv('FS_SCORE_METHOD', 'affinity').lower()
            self.used_cross_val_method = os.getenv('CROSS_VAL').lower()
            self.tf_idf_config = os.getenv('TF_IDF_CONFIG').lower()
            self.number_of_splits = os.getenv('NUMBER_OF_SPLITS').lower()
            self.seed = os.getenv('SEED').lower()

            # GridSearch default params
            self.gs_cv = int(os.getenv('GS_CV', 5))
            self.gs_scoring = os.getenv('GS_SCORING', 'f1').lower()

            self.env_spec = [
                ['DATASET_TYPE', self.used_dataset],
                ['USED_FEATURE_SELECTION', self.used_fs],
                ['FS_SCORE_METHOD', self.used_score_method],
                ['CROSS_VAL', self.used_cross_val_method],
                ['TF_IDF_CONFIG', self.tf_idf_config],
                ['NUMBER_OF_SPLITS', self.number_of_splits],
                ['SEED', self.seed],
                ['ENV_FILE', self.env_file_path],
                ['GS_CV', self.gs_cv],
                ['GS_SCORING', self.gs_scoring],
            ]

            print('\n\t TEST CONFIGURATION FROM ENV FILE:')
            for k, v in self.env_spec:
                print('\t\t{} = {}'.format(k, v))
        else:
            raise FileNotFoundError

    def get_dataset_type(self):
        return DATASET_TYPES[self.used_dataset]

    def get_score_method_type(self):
        return FS_SCORE_METHODS[self.used_score_method]

    def get_cross_val_type(self):
        return CROSS_VALS[self.used_cross_val_method]

    def use_feature_selection(self):
        return self.used_fs

    def get_env_vars_spec(self):
        return self.env_spec

    def get_grid_search_params(self):
        return {'gs_cv': self.gs_cv, 'gs_scoring': self.gs_scoring}

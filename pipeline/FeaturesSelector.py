# Author: Marcelo Costalonga
# (code adapted from: https://github.com/watinha/automatic-selection-slr/tree/master/pipeline/feature_selection)

from TestConfigurationLoader import TestConfiguration
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif, r_regression
import pandas as pd


# Class to apply Feature Selection technique over a vectorized entry to determinate the best kN features
class FeaturesSelector:
    df_all_set = dict()
    training_dataset = None
    testing_dataset = None

    def __init__ (self, k_fs=-1):
        try:
            assert k_fs > 0
        except AssertionError:
            print("\nMissing number of k features for Feature Selector.")
            exit(0)
        self._k = k_fs
        self._affinity_score = []

    def _affinity (self, word_frequency_column, categories, category):
        ncw = 0
        nc = 0
        nw = 0
        for i in range(0, len(categories)):
            if categories[i] == category: nc += 1
            if word_frequency_column[i] > 0: nw += 1
            if categories[i] == category and word_frequency_column[i] > 0: ncw += 1
        return ncw / (nc + nw - ncw)

    def _score (self, features, categories):
        n_words = features.shape[1]
        self._affinity_score = [
            self._affinity(features[:, i], categories, 1) -
            self._affinity(features[:, i], categories, 0)
            for i in range(0, n_words)
        ]
        return (self._affinity_score, [])

    def _split_into_dataframes(self, dataset: dict, year_to_split: int):
        df = pd.DataFrame.from_dict(dataset)

        # Create a mask to filter the bibs published before (training) and after (testing) the year of update
        train_mask = df['years'] <= year_to_split
        test_mask = df['years'] > year_to_split

        # Split the data into training and testing sets
        df_train = df[train_mask]
        df_test = df[test_mask]

        # TODO: improve this object
        return [df_train, df_test]

    def execute(self, training_dataset: dict, testing_dataset: dict):
        print('===== Feature selection - Selecting {} best features ====='.format(self._k))
        # TODO#: Single dataset vs Two datasets
        # X = dataset['features']
        # y = dataset['categories']

        X_train = training_dataset['features']
        y_train = training_dataset['categories']
        X_test = testing_dataset['features']


        # Loading dataset configuration
        used_score_method = TestConfiguration().get_score_method_type()

        # FIXME #3: Check different configurations for SelectKBest (maybe chi2? SelectKBest(chi2, k=1000))
        # FIXME #4: This could improve the time, 'fs.fit_transform' its a huge bottle neck that takes a lot of time to process

        # Initialze selector
        if used_score_method == 0:
            # Default affinity method used
            fs = SelectKBest(self._score, k=self._k)
        elif used_score_method == 1:  # Chi2
            fs = SelectKBest(chi2, k=self._k)
        elif used_score_method == 2:  # Anova F
            fs = SelectKBest(f_classif, k=self._k)
        elif used_score_method == 3:  # Pearson Correlation
            fs = SelectKBest(r_regression, k=self._k)
        else:
            print("\n[ERROR-EnvFile] Invalid feature selection score method option")
            raise Exception


        # Fit using only the training set
        # FIXME#11!: Add check to verify if K is <= to the total number of features (if it is read again the input)
        #  (different K's for test and training???)
        fs.fit(X_train, y_train)
        training_dataset['features'] = fs.transform(X_train) # transform training set
        testing_dataset['features'] = fs.transform(X_test) # transform testing set

        self.training_dataset = training_dataset
        self.testing_dataset = testing_dataset

        # FIXME #0: Check if spliting by year would be relevant for cross-validation... (change to avoid using hardcoded  2014 year)
        # tmp_set = self._split_into_dataframes(dataset, year_to_split=2014)
        # self.df_training_set = tmp_set[0]
        # self.df_testing_set = tmp_set[1]
        # self.df_all_set = dataset # TODO#: Single dataset vs Two datasets
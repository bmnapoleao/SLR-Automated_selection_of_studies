# Author: Marcelo Costalonga
# (code adapted from: https://github.com/watinha/automatic-selection-slr/tree/master/pipeline/feature_selection)

from sklearn.feature_selection import SelectKBest

# Class to apply Feature Selection technique over a vectorized entry to determinate the best kN features
class FeaturesSelector:
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
            self._affinity(features[:,i], categories, 1) -
            self._affinity(features[:,i], categories, 0)
            for i in range(0, n_words) ]
        return (self._affinity_score, [])

    def execute(self, dataset):
        print('===== Feature selection - Selecting {} best features ====='.format(self._k))
        X = dataset['features']
        y = dataset['categories']
        fs = SelectKBest(self._score, k=self._k)
        dataset['features'] = fs.fit_transform(X, y)
        return dataset

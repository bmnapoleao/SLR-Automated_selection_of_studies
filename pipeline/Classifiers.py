# Author: Marcelo Costalonga
# (code adapted from: https://github.com/watinha/automatic-selection-slr/blob/master/pipeline/classifiers/__init__.py)

import random
from sklearn import tree, svm
from sklearn.model_selection import cross_validate, GridSearchCV

# Class to split content of training set into multiple folds, grouping them by a specific range of years.
class YearsSplit:
    def __init__ (self, n_splits=4, years=[]):
        self._n_splits = n_splits
        self._years = years
        self._test_indexes = []
        current = max(years)
        for i in range(n_splits):
            test_index = years.index(current)
            if len(years[test_index:]) < 5:
                current = max(years[:test_index])
                test_index = years.index(current)

            self._test_indexes.append(test_index)
            current = max(years[:test_index])

    def split (self, X, y, groups=None):
        previous = len(self._years)
        for test_index in self._test_indexes:
            train = [ i for i in range(test_index) ]
            test = [ i for i in range(test_index, previous) ]
            previous = test_index
            yield train, test

# Simple model of an ML classifier (parent class used for other classifiers).
# Trains the model applying the holdout technique, using the training dataset and reports its results
class SimpleClassifier:
    def __init__(self, seed, n_splits=5):
        self._seed = seed
        self._n_splits = n_splits

    def execute(self, dataset_train, dataset_test):
        print("Executing...")
        X = dataset_train['features']
        y = dataset_train['categories']

        groups = dataset_train['years']
        random.seed(self._seed)

        kfold = YearsSplit(n_splits=self._n_splits, years=groups)
        model = self.get_classifier(X, y)  # Select either DT or SVM classifier model

        scores = cross_validate(model, X, y, cv=kfold, scoring=['f1_macro', 'precision_macro', 'recall_macro'], error_score="raise")
        print("OUR APPROACH F-measure: %s on average and %s SD" %
              (scores['test_f1_macro'].mean(), scores['test_f1_macro'].std()))
        print("OUR APPROACH Precision: %s on average and %s SD" %
              (scores['test_precision_macro'].mean(), scores['test_precision_macro'].std()))
        print("OUR APPROACH Recall: %s on average and %s SD" %
              (scores['test_recall_macro'].mean(), scores['test_recall_macro'].std()))

        X_train = dataset_train['features']
        y_train = dataset_train['categories']
        X_test = dataset_test['features']
        y_test = dataset_test['categories']

        model.fit(X_train, y_train)

        # Perform test and get prediction
        y_pred = model.predict(X_test)

        return y_pred

# Extends the SimpleClassifier class to use the DecisionTree classifier with a specific configuration
class DecisionTreeClassifier (SimpleClassifier):
    def __init__ (self, seed=42, criterion='entropy', n_splits=5):
        SimpleClassifier.__init__(self, seed, n_splits)
        self.classifier_name = 'decision_tree'
        self._criterion = criterion

    def get_classifier (self, X, y):
        print('===== Decision Tree Classifier =====')
        print('===== Hyperparameter tunning  =====')
        # model = tree.DecisionTreeClassifier()
        # params = {
        #     'criterion': ["gini", "entropy"],
        #     'max_depth': [10, 50, 100, None],
        #     'min_samples_split': [2, 10, 100],
        #     'class_weight': [None, 'balanced']
        # }
        # cfl = GridSearchCV(model, params, cv=5, scoring='recall')
        # cfl.fit(X, y)
        # for param, value in cfl.best_params_.items():
        #     print("%s : %s" % (param, value))

        model = tree.DecisionTreeClassifier(random_state=self._seed)
        # model.set_params(**cfl.best_params_)
        return model

# Extends the SimpleClassifier class to use the SVM classifier with a specific configuration
class SVMClassifier (SimpleClassifier):
    def __init__ (self, seed=42, n_splits=3):
        SimpleClassifier.__init__(self, seed, n_splits=n_splits)
        self.classifier_name = 'svm'

    def get_classifier (self, X, y):
        print('===== SVM Classifier =====')
        print('===== Hyperparameter tunning  =====')
        # params = {
        #     'kernel': ['linear', 'rbf'],
        #     'C': [1, 10, 100],
        #     'tol': [0.001, 0.1, 1],
        #     'class_weight': ['balanced', None]
        # }
        # model = svm.SVC(random_state=self._seed, probability=True)
        # cfl = GridSearchCV(model, params, cv=5, scoring='accuracy')
        # cfl.fit(X, y)
        # for param, value in cfl.best_params_.items():
        #     print("%s : %s" % (param, value))
        model = svm.SVC(random_state=self._seed, probability=True)
        # model.set_params(**cfl.best_params_)
        return model

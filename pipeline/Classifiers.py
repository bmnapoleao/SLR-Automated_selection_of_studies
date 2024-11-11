# Author: Marcelo Costalonga
# (code adapted from: https://github.com/watinha/automatic-selection-slr/blob/master/pipeline/classifiers/__init__.py)

import random
from sklearn import metrics
from sklearn.metrics import accuracy_score
# from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.metrics import precision_score, recall_score, f1_score
from TestConfigurationLoader import TestConfiguration
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier
from sklearn.model_selection import cross_validate, GridSearchCV, TimeSeriesSplit
# from sklearn.naive_bayes import GaussianNB
# from sklearn.linear_model import LinearRegression, LogisticRegression, LinearRegression


# Class to split content of training set into multiple folds, grouping them by a specific range of years.
class YearsSplit:
    def __init__(self, n_splits=4, years=[]):
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
    classifier_name = ''
    classifier_label = ''
    best_params = dict()
    grid_search_tested_params = dict()

    def __init__(self, seed, n_splits=5):
        self._seed = seed
        self._n_splits = n_splits
        self._cross_val_method = TestConfiguration().get_cross_val_type()

    def execute(self, training_set: dict, testing_set: dict):
        X_train = training_set['features']
        y_train = training_set['categories']

        X_test = testing_set['features']
        y_test = testing_set['categories']

        # Get classifier model
        model = self.get_classifier(X_train, y_train)

        cross_val_scores = dict()

        # kfold cross validation (splited by years)
        if self._cross_val_method == 0:
            groups = training_set['years']
            random.seed(self._seed)
            kfold = YearsSplit(n_splits=self._n_splits, years=groups)
            cross_val_scores = cross_validate(model, X_train, y_train, cv=kfold, scoring=['f1_macro', 'precision_macro', 'recall_macro'])
            model.fit(X_train, y_train)

        # time series cross validation
        elif self._cross_val_method == 1:
            threasholds = []
            fscore_threashold = []

            # set up time series cross-validator
            tscv = TimeSeriesSplit(n_splits=self._n_splits)

            for train_index, test_index in tscv.split(X_train, y_train):
                X_train_index, X_test_index = X_train[train_index], X_train[test_index]
                y_train_index, y_test_index = y_train[train_index], y_train[test_index]
                model.fit(X_train_index, y_train_index)
                y_score_index = model.predict_proba(X_train_index)[:, 1]
                precision, recall, threasholds2 = metrics.precision_recall_curve(y_train_index, y_score_index)
                y_score_index = model.predict_proba(X_test_index)[:, 1]
                if (threasholds2[0] > 0.5):
                    threasholds2 = [0.5]

                threasholds.append(threasholds2[0])
                fscore_threashold.append(metrics.f1_score(
                    y_test_index, [0 if i < threasholds2[0] else 1 for i in y_score_index]))

            cross_val_scores['threasholds_cros_val?'] = threasholds
            cross_val_scores['fscore_threashold_cros_val?'] = fscore_threashold
            print(f"threasholds: {cross_val_scores['threasholds_cros_val?']}")
            print(f"fscore_threashold: {cross_val_scores['fscore_threashold_cros_val?']}")

        # without applying cross-validation because we're already using GridSearch
        elif self._cross_val_method == 2:
            pass

        else:
            print("\n[ERROR-EnvFile] Invalid cross validation method")
            raise Exception

        # Perform prediction test
        predictions = dict()
        y_pred = model.predict(X_test)
        predictions['y_pred'] = y_pred
        predictions['y_proba'] = model.predict_proba(X_test)[:, 1]  # Only the prob of being 1 (selected)

        # compute the metrics for the test set
        scores = dict()
        scores['accuracy'] = accuracy_score(y_test, y_pred)
        scores['precision'] = precision_score(y_test, y_pred)
        scores['recall'] = recall_score(y_test, y_pred)
        scores['F1'] = f1_score(y_test, y_pred)

        # Print test scores
        print('Classifier Test Metrics:')
        print(f"Accuracy: {scores['accuracy']}")
        print(f"Precision: {scores['precision']}")
        print(f"Recall: {scores['recall']}")
        print(f"F1: {scores['F1']}")
        print("-------------------------------------------------------------------------------------------------------\n\n")
        return {'predictions': predictions, 'scores': scores,
                'best_params': self.best_params, 'tested_params': self.grid_search_tested_params}


# Extends the SimpleClassifier class to use the DecisionTree classifier with a specific configuration
class DecisionTreeClassifier (SimpleClassifier):
    def __init__(self, seed=42, criterion='entropy', n_splits=5):
        SimpleClassifier.__init__(self, seed, n_splits)
        self.classifier_name = 'decision_tree'
        self.classifier_label = 'dt'
        self._criterion = criterion

    def get_classifier(self, X, y):
        print('\n\n===== Decision Tree Classifier ===== \n\t n_splits=', self._n_splits)
        print('===== Hyperparameter tuning (best params) =====')
        model = SklearnDecisionTreeClassifier()
        # If cross_validation different from GridSearch skip it so we don't perform cross_validation more than once
        if self._cross_val_method != 2:
            return model

        params = {
            'criterion': ["gini", "entropy"],
            'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
            'min_samples_split': [2, 10, 25, 50, 75, 100],
            'class_weight': [None, 'balanced']
        }

        grid_search_default_params = TestConfiguration().get_grid_search_params()
        gs_cv = grid_search_default_params['gs_cv']
        gs_scoring = grid_search_default_params['gs_scoring']

        grid_search = GridSearchCV(model, params, cv=gs_cv, scoring=gs_scoring)
        grid_search.fit(X, y)
        for param, value in grid_search.best_params_.items():
            print("%s : %s" % (param, value))
        print("-----------------------------------------------\n\n")
        self.best_params = grid_search.best_params_
        self.grid_search_tested_params = params
        self.grid_search_tested_params['cv'] = gs_cv
        self.grid_search_tested_params['scoring'] = gs_scoring
        model = grid_search.best_estimator_
        return model


# Extends the SimpleClassifier class to use the SVM classifier with a specific configuration
class SVMClassifier (SimpleClassifier):
    def __init__(self, seed=42, n_splits=3):
        SimpleClassifier.__init__(self, seed, n_splits=n_splits)
        self.classifier_name = 'support_vector_machine'
        self.classifier_label = 'svm'

    def get_classifier(self, X, y):
        print('\n\n===== SVM Classifier ===== \n\t n_splits=', self._n_splits)
        model = SVC(random_state=self._seed, probability=True)
        # If cross_validation different from GridSearch skip it so we don't perform cross_validation more than once
        if self._cross_val_method != 2:
            return model

        print('===== Hyperparameter tuning (best params) =====')
        params = {
            'kernel': ['rbf', 'sigmoid'],
            'C': [0.5, 0.65, 0.8, 0.95, 1.1, 1.25, 1.4, 1.55, 1.7, 1.85, 2.0, 2.15, 2.3, 2.45, 2.6, 2.75, 2.9, 3.05,
                  3.2, 3.35],
            'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
            'tol': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85,
                    0.9, 0.95, 1.0],
            'class_weight': ['balanced', None]
        }

        # # Best params for RQ2
        # params = {
        #     'kernel': ['rbf'],
        #     'C': [0.65],
        #     'gamma': [1],
        #     'tol': [0.8],
        #     'class_weight': ['balanced']
        # }

        grid_search_default_params = TestConfiguration().get_grid_search_params()
        gs_cv = grid_search_default_params['gs_cv']
        gs_scoring = grid_search_default_params['gs_scoring']
        grid_search = GridSearchCV(model, params, cv=gs_cv, scoring=gs_scoring)

        grid_search.fit(X, y)
        for param, value in grid_search.best_params_.items():
            print("%s : %s" % (param, value))
        print("-----------------------------------------------\n\n")
        self.best_params = grid_search.best_params_
        self.grid_search_tested_params = params
        self.grid_search_tested_params['cv'] = gs_cv
        self.grid_search_tested_params['scoring'] = gs_scoring
        model = grid_search.best_estimator_
        return model


# Extends the SimpleClassifier class to use the KNeighbors classifier with a specific configuration
class KNNClassifier (SimpleClassifier):
    def __init__(self, seed=42, n_splits=3):
        SimpleClassifier.__init__(self, seed, n_splits=n_splits)
        self.classifier_name = ' k_nearest_neighbors'
        self.classifier_label = 'knn'

    def get_classifier(self, X, y):
        print('\n\n===== KNN Classifier ===== \n\t n_splits=', self._n_splits)
        print('===== Hyperparameter tuning (best params) =====')
        model = KNeighborsClassifier()
        # If cross_validation different from GridSearch skip it so we don't perform cross_validation more than once
        if self._cross_val_method != 2:
            return model

        params = {
            'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
            'metric': ['euclidean', 'manhattan', 'minkowski'],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree'],
        }

        grid_search_default_params = TestConfiguration().get_grid_search_params()
        gs_cv = grid_search_default_params['gs_cv']
        gs_scoring = grid_search_default_params['gs_scoring']

        cfl = GridSearchCV(model, params, cv=gs_cv, scoring=gs_scoring)
        self.grid_search_tested_params['cv'] = gs_cv
        self.grid_search_tested_params['scoring'] = gs_scoring
        cfl.fit(X, y)
        for param, value in cfl.best_params_.items():
            print("%s : %s" % (param, value))
        print("-----------------------------------------------\n\n")
        self.best_params = cfl.best_params_
        model = KNeighborsClassifier()
        model.set_params(**cfl.best_params_)
        return model


class RandomForestClassifier (SimpleClassifier):
    def __init__(self, seed=42, n_splits=5):
        SimpleClassifier.__init__(self, seed, n_splits)
        self.classifier_name = 'random_forest'
        self.classifier_label = 'rforest'

    def get_classifier(self, X, y):
        print('\n\n===== Random Forest Classifier ===== \n\t n_splits=', self._n_splits)
        print('===== Hyperparameter tuning (best params) =====')
        model = SklearnRandomForestClassifier(random_state=self._seed)
        # If cross_validation different from GridSearch skip it so we don't perform cross_validation more than once
        if self._cross_val_method != 2:
            return model

        params = {
            # 'n_estimators': [100, 200, 300],
            'n_estimators': [2, 5, 100],
            # 'max_features': ["sqrt", "auto"],
            'criterion': ["gini", "entropy"],
            # 'max_depth':[10, 25, 50, 75, 100, None],
            'max_depth':[10, 20, 30, None],
            'min_samples_split': [2, 10, 25, 50, 75, 100],
            'class_weight': [None, 'balanced']
        }

        # # Best params for RQ1
        # params = {
        #     'n_estimators': [100],
        #     'criterion': ["gini"],
        #     'max_depth':[10],
        #     'min_samples_split': [10],
        #     'class_weight': ['balanced']
        # }

        grid_search_default_params = TestConfiguration().get_grid_search_params()
        gs_cv = grid_search_default_params['gs_cv']
        gs_scoring = grid_search_default_params['gs_scoring']

        cfl = GridSearchCV(model, params, cv=gs_cv, scoring=gs_scoring)
        cfl.fit(X, y)
        for param, value in cfl.best_params_.items():
            print("\t%s : %s" % (param, value))

        self.best_params = cfl.best_params_
        self.grid_search_tested_params = params
        self.grid_search_tested_params['cv'] = gs_cv
        self.grid_search_tested_params['scoring'] = gs_scoring
        model = cfl.best_estimator_
        return model


# # Extends the SimpleClassifier class to use the Gaussian Naive Bayes classifier with a specific configuration
# class GaussianNaiveBayesClassifier (SimpleClassifier):
#     def __init__(self, seed=42, n_splits=3):
#         SimpleClassifier.__init__(self, seed, n_splits=n_splits)
#         self.classifier_name = 'gaussian_naive_bayes'
#         self.classifier_label = 'gaussianNB'

#     def get_classifier(self, X, y):
#         print('\n\n===== GaussianNB Classifier =====')
#         model = GaussianNB()
#         return model


# # Extends the SimpleClassifier class to use the Linear Regression classifier with a specific configuration
# class LinearRegressionClassifier (SimpleClassifier):
#     def __init__(self, seed):
#         SimpleClassifier.__init__(self, seed)
#         self.classifier_name = 'linear_regression'
#         self.classifier_label = 'lin_reg'

#     def get_classifier(self, X, y):
#         print('===== Linear Reg Classifier =====')
#         return LinearRegression()


# # Extends the SimpleClassifier class to use the Logistic Regression classifier with a specific configuration
# class LogisticRegressionClassifier (SimpleClassifier):
#     def __init__(self, seed):
#         SimpleClassifier.__init__(self, seed)
#         self.classifier_name = 'logistic_regression'
#         self.classifier_label = 'log_reg'

#     def get_classifier(self, X, y):
#         print('===== Logistic Reg Classifier =====')
#         return LogisticRegression(random_state=self._seed)

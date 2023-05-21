# Author: Marcelo Costalonga
# (code adapted from: https://github.com/watinha/automatic-selection-slr/blob/master/pipeline/classifiers/__init__.py)

import random
import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.metrics import precision_score, recall_score, f1_score
from TestConfigurationLoader import TestConfiguration

# Classifiers
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV, TimeSeriesSplit, train_test_split


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

    def __init__(self, seed, n_splits=5, parameters_config=None):
        self._seed = seed
        self._n_splits = n_splits
        self._config_used = parameters_config

    def execute(self, training_set: dict, testing_set: dict):
        X_train = training_set['features']
        y_train = training_set['categories']

        X_test = testing_set['features']
        y_test = testing_set['categories']

        # Get classifier model
        model = self.get_classifier(X_train, y_train)

        # Loading dataset configuration
        used_cross_val_method = TestConfiguration().get_cross_val_type()

        cross_val_scores = dict()

        # kfold cross validation (splited by years)
        if used_cross_val_method == 0:
            # FIXME#10: If we're going to use this approach should it be all years (including testing)?
            groups = training_set['years']
            random.seed(self._seed)
            kfold = YearsSplit(n_splits=self._n_splits, years=groups)

            # FIXME#22: Start keeping track of performance from cross validation
            # scores = cross_validate(model, X_train, y_train, cv=kfold, scoring=['f1_macro', 'precision_macro', 'recall_macro'])
            cross_val_scores = cross_validate(model, X_train, y_train, cv=kfold,
                                              scoring=['f1_macro', 'precision_macro', 'recall_macro'])
            print("OUR APPROACH F-measure: %s on average and %s SD" %
                  (cross_val_scores['test_f1_macro'].mean(), cross_val_scores['test_f1_macro'].std()))
            print("OUR APPROACH Precision: %s on average and %s SD" %
                  (cross_val_scores['test_precision_macro'].mean(), cross_val_scores['test_precision_macro'].std()))
            print("OUR APPROACH Recall: %s on average and %s SD" %
                  (cross_val_scores['test_recall_macro'].mean(), cross_val_scores['test_recall_macro'].std()))
            print("-----------------------------------------------------------\n")

            model.fit(X_train, y_train)

        # time series cross validation
        elif used_cross_val_method == 1:
            threasholds = []
            fscore_threashold = []

            # set up time series cross-validator
            tscv = TimeSeriesSplit(n_splits=self._n_splits)

            # FIXME#7: Understand if it makes sense to use YearsSplit method
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
            # scores['exclusion_rate'] = correct_exclusion_rate
            # scores['missed'] = missed
            # scores['exclusion_baseline'] = exclusion_baseline
            # scores['missed_baseline'] = missed_baseline

            # Print the results
            # print(f"exclusion_rate: {scores['exclusion_rate']}")
            # print(f"missed: {scores['missed']}")
            # print(f"exclusion_baseline: {scores['exclusion_baseline']}")
            # print(f"missed_baseline: {scores['missed_baseline']}")
            print(f"threasholds: {cross_val_scores['threasholds_cros_val?']}")
            print(f"fscore_threashold: {cross_val_scores['fscore_threashold_cros_val?']}")

        else:
            print("\n[ERROR-EnvFile] Invalid cross validation method")
            raise Exception

        # Perform prediction test
        predictions = dict()
        y_pred = model.predict(X_test)

        predictions['y_pred'] = y_pred

        # FIXME#24: Understand why DT model predict_proba always 0 or 1
        #   OBS: Problema de 0 1 da DT: https://stackoverflow.com/questions/48219986/decisiontreeclassifier-predict-proba-returns-0-or-1
        predictions['y_proba'] = model.predict_proba(X_test)[:, 1]  # Only the prob of being 1 (selected)

        # compute the metrics for the test set
        scores = dict()
        scores['accuracy'] = accuracy_score(y_test, y_pred)
        scores['precision'] = precision_score(y_test, y_pred)
        scores['recall'] = recall_score(y_test, y_pred)
        scores['F1'] = f1_score(y_test, y_pred)

        # scores['conf_matrix'] = confusion_matrix(y_test, y_pred_test) # FIXME#14: Storage and report confusion matrix in another way

        # Print test scores
        print('Classifier Test Metrics:')
        print(f"Accuracy: {scores['accuracy']}")
        print(f"Precision: {scores['precision']}")
        print(f"Recall: {scores['recall']}")
        print(f"F1: {scores['F1']}")

        # FIXME#14: Investigate error message: "UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
        #   _warn_prf(average, modifier, msg_start, len(result))" OBS: only shows on pycharm, executing script from terminal does not display this message

        # return ClassifierExecutionResult(predictions, scores, self.best_params)
        print("-------------------------------------------------------------------------------------------------------\n\n")
        return {'predictions': predictions, 'scores': scores, 'best_params': self.best_params}


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
        params = {
            'criterion': ["gini", "entropy"],
            'max_depth': [10, 50, 100, None],
            'min_samples_split': [2, 10, 100],
            'class_weight': [None, 'balanced']
        }
        cfl = GridSearchCV(model, params, cv=5, scoring='accuracy')
        cfl.fit(X, y)
        for param, value in cfl.best_params_.items():
            print("%s : %s" % (param, value))
        print("-----------------------------------------------\n\n")
        self.best_params = cfl.best_params_
        model = SklearnDecisionTreeClassifier(random_state=self._seed)
        model.set_params(**cfl.best_params_)
        return model


# Extends the SimpleClassifier class to use the SVM classifier with a specific configuration
class SVMClassifier (SimpleClassifier):
    def __init__(self, seed=42, n_splits=3):
        SimpleClassifier.__init__(self, seed, n_splits=n_splits)
        self.classifier_name = 'support_vector_machine'
        self.classifier_label = 'svm'

    def get_classifier(self, X, y):
        print('\n\n===== SVM Classifier ===== \n\t n_splits=', self._n_splits)
        print('===== Hyperparameter tuning (best params) =====')
        params = {
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'C': [0.001, 0.005, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0, 2.4, 2.6, 2.8, 3.0, 3.3, 3.5, 3.7,
                  4.0, 10, 100, 1000],
            'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
            'tol': [0.0001, 0.001, 0.01, 0.1, 1],
            'class_weight': ['balanced', None]
        }
        model = SVC(random_state=self._seed, probability=True)
        cfl = GridSearchCV(model, params, cv=5, scoring='accuracy')
        cfl.fit(X, y)
        for param, value in cfl.best_params_.items():
            print("%s : %s" % (param, value))
        print("-----------------------------------------------\n\n")
        self.best_params = cfl.best_params_
        model = SVC(random_state=self._seed, probability=True)
        model.set_params(**cfl.best_params_)
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
        params = {
            'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
            'metric': ['euclidean', 'manhattan', 'minkowski'],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree'],
        }
        model = KNeighborsClassifier()
        cfl = GridSearchCV(model, params, cv=5, scoring='accuracy')
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
        params = {
            'n_estimators': [5, 10, 100],
            'criterion': ["gini", "entropy"],
            'max_depth': [10, 50, 100, None],
            'min_samples_split': [2, 10, 100],
            'class_weight': [None, 'balanced']
        }
        cfl = GridSearchCV(model, params, cv=5, scoring='accuracy')
        cfl.fit(X, y)
        for param, value in cfl.best_params_.items():
            print("\t%s : %s" % (param, value))

        self.best_params = cfl.best_params_
        model = SklearnRandomForestClassifier(random_state=self._seed)
        model.set_params(**cfl.best_params_)
        return model


# Extends the SimpleClassifier class to use the Gaussian Naive Bayes classifier with a specific configuration
class GaussianNaiveBayesClassifier (SimpleClassifier):
    def __init__(self, seed=42, n_splits=3):
        SimpleClassifier.__init__(self, seed, n_splits=n_splits)
        self.classifier_name = 'gaussian_naive_bayes'
        self.classifier_label = 'gaussianNB'

    def get_classifier(self, X, y):
        print('\n\n===== GaussianNB Classifier =====')
        model = GaussianNB()
        return model


# Extends the SimpleClassifier class to use the Linear Regression classifier with a specific configuration
class LinearRegressionClassifier (SimpleClassifier):
    def __init__(self, seed):
        SimpleClassifier.__init__(self, seed)
        self.classifier_name = 'linear_regression'
        self.classifier_label = 'lin_reg'

    def get_classifier(self, X, y):
        print('===== Linear Reg Classifier =====')
        return LinearRegression()


# Extends the SimpleClassifier class to use the Logistic Regression classifier with a specific configuration
class LogisticRegressionClassifier (SimpleClassifier):
    def __init__(self, seed):
        SimpleClassifier.__init__(self, seed)
        self.classifier_name = 'logistic_regression'
        self.classifier_label = 'log_reg'

    def get_classifier(self, X, y):
        print('===== Logistic Reg Classifier =====')
        return LogisticRegression(random_state=self._seed)

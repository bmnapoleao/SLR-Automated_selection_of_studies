# Author: Marcelo Costalonga
# (code adapted from: https://github.com/watinha/automatic-selection-slr/blob/master/pipeline/classifiers/__init__.py)

import random

import numpy as np
from sklearn import tree, svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate, GridSearchCV, TimeSeriesSplit
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest

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

    def execute(self, df_training_set: pd.DataFrame, df_testing_set: pd.DataFrame, fs: SelectKBest):
        print("Executing...")

        X_train = df_training_set['features']  # Drop the 'category' column for the input features
        y_train = df_training_set['categories']
        X_test = df_testing_set['features']
        y_test = df_testing_set['categories']

        # TODO: compare
        # groups = df_training_set['years']
        # random.seed(self._seed)
        # kfold = YearsSplit(n_splits=self._n_splits, years=groups)

        # set up time series cross-validator
        tscv = TimeSeriesSplit(n_splits=5)

        # initialize a list to store the metrics for each fold
        fold_accuracies = []
        fold_precisions = []
        fold_recalls = []
        fold_f_measures = []

        # Get classifier model (DT or SVM)
        model = self.get_classifier(X_train, y_train)

        # loop through each fold and train/test the model
        for train_index, test_index in tscv.split(X_train):
            # select the data for the current fold
            X_train_fold = X_train.iloc[train_index].values
            y_train_fold = y_train.iloc[train_index].values
            X_train_fold = fs.transform(X_train_fold)
            X_train_fold = X_train_fold.toarray()

            X_val_fold = X_train.iloc[test_index].values
            y_val_fold = y_train.iloc[test_index].values

            # Train the model on the training data
            model.fit(X_train_fold, y_train_fold)

            # test the model on the testing data
            y_pred_fold = model.predict(X_val_fold)

            # compute the metrics for the fold
            accuracy_fold = accuracy_score(y_val_fold, y_pred_fold)
            precision_fold, recall_fold, f_measure_fold, _ = classification_report(y_val_fold, y_pred_fold, output_dict=True)['weighted avg']

            # store the metrics for the fold
            fold_accuracies.append(accuracy_fold)
            fold_precisions.append(precision_fold)
            fold_recalls.append(recall_fold)
            fold_f_measures.append(f_measure_fold)

        # calculate the mean metrics across all folds
        mean_accuracy = np.mean(fold_accuracies)
        mean_precision = np.mean(fold_precisions)
        mean_recall = np.mean(fold_recalls)
        mean_f_measure = np.mean(fold_f_measures)

        # TODO: check if it's needed to fit the model again or not
        # fit the final model on the full training set and test it on the testing set
        # model.fit(X_train, y_train)

        # perform the test
        y_pred_test = model.predict(X_test)

        # compute the metrics for the test set
        accuracy_test = accuracy_score(y_test, y_pred_test)
        precision_test, recall_test, f_measure_test, _ = classification_report(y_test, y_pred_test, output_dict=True)[
            'weighted avg']

        # print the results
        print(f"Mean cross-validation accuracy: {mean_accuracy}")
        print(f"Mean cross-validation precision: {mean_precision}")
        print(f"Mean cross-validation recall: {mean_recall}")
        print(f"Mean cross-validation F-measure: {mean_f_measure}")
        print(f"Test set accuracy: {accuracy_test}")
        print(f"Test set precision: {precision_test}")
        print(f"Test set recall: {recall_test}")
        print(f"Test set F-measure: {f_measure_test}")

        # Returns the y_test with for the currently model
        return y_pred_test

    # def execute(self, dataset_train, dataset_test):
    #     print("Executing...")
    #     X = dataset_train['features']
    #     y = dataset_train['categories']
    #
    #     groups = dataset_train['years']
    #     random.seed(self._seed)
    #
    #     kfold = YearsSplit(n_splits=self._n_splits, years=groups)
    #     model = self.get_classifier(X, y)  # Select either DT or SVM classifier model
    #
    #     scores = cross_validate(model, X, y, cv=kfold, scoring=['f1_macro', 'precision_macro', 'recall_macro'], error_score="raise")
    #     print("OUR APPROACH F-measure: %s on average and %s SD" %
    #           (scores['test_f1_macro'].mean(), scores['test_f1_macro'].std()))
    #     print("OUR APPROACH Precision: %s on average and %s SD" %
    #           (scores['test_precision_macro'].mean(), scores['test_precision_macro'].std()))
    #     print("OUR APPROACH Recall: %s on average and %s SD" %
    #           (scores['test_recall_macro'].mean(), scores['test_recall_macro'].std()))
    #
    #     X_train = dataset_train['features']
    #     y_train = dataset_train['categories']
    #     X_test = dataset_test['features']
    #     y_test = dataset_test['categories']
    #
    #     model.fit(X_train, y_train)
    #
    #     # Perform test and get prediction
    #     y_pred = model.predict(X_test)
    #
    #     return y_pred

# Extends the SimpleClassifier class to use the DecisionTree classifier with a specific configuration
class DecisionTreeClassifier (SimpleClassifier):
    def __init__ (self, seed=42, criterion='entropy', n_splits=5):
        SimpleClassifier.__init__(self, seed, n_splits)
        self.classifier_name = 'decision_tree'
        self._criterion = criterion

    def get_classifier (self, X, y):
        print('===== Decision Tree Classifier =====')
        print('===== Hyperparameter tunning  =====')
        model = tree.DecisionTreeClassifier()

        # params = {
        #     'criterion': ["gini", "entropy"],
        #     'max_depth': [10, 50, 100, None],
        #     'min_samples_split': [2, 10, 100],
        #     'class_weight': [None, 'balanced']
        # }
        #
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

# Author: Marcelo Costalonga
# (code adapted from: https://github.com/watinha/automatic-selection-slr/blob/master/pipeline/classifiers/__init__.py)

import random

import numpy as np
from sklearn import tree, metrics, svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate, GridSearchCV, TimeSeriesSplit, train_test_split
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

    def execute(self, dataset: dict(), fs: SelectKBest):
        print("Executing...")

        # X = dataset  # Drop the 'category' column for the input features
        # y = X.pop('categories') # Drop the 'category' column for the input features

        # df = pd.DataFrame(dataset)
        # X = df['features']  # Drop the 'category' column for the input features
        # y = df['categories']

        X = dataset['features']  # Drop the 'category' column for the input features
        y = dataset['categories']

        groups = dataset['years']
        random.seed(self._seed)
        kfold = YearsSplit(n_splits=self._n_splits, years=groups)
        # Get classifier model (DT or SVM)
        model = self.get_classifier(X, y)

        scores = cross_validate(model, X, y, cv=kfold, scoring=['f1_macro', 'precision_macro', 'recall_macro'])
        print("OUR APPROACH F-measure: %s on average and %s SD" %
              (scores['test_f1_macro'].mean(), scores['test_f1_macro'].std()))
        print("OUR APPROACH Precision: %s on average and %s SD" %
              (scores['test_precision_macro'].mean(), scores['test_precision_macro'].std()))
        print("OUR APPROACH Recall: %s on average and %s SD" %
              (scores['test_recall_macro'].mean(), scores['test_recall_macro'].std()))

        # initialize a list to store the metrics for each fold
        fold_accuracies = []
        fold_precisions = []
        fold_recalls = []
        fold_f_measures = []

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


        # Get the names of the rows in X_test
        # test_names = df.iloc[X_test.index]['texts']
        # test_years = df.iloc[X_test.index]['years']

        # FIXME#3: Verificar como obter o nome de cada artigo referente ao testing set
        # test_names = dataset['texts']
        # test_years = dataset['years']
        #
        # print("\n\n\nTESTING:")
        # for name, year in zip(test_names, test_years):
        #     print("{} - {}".format(year, name))
        # print(":TESTING\n\n\n")



        # FIXME#6: Predict at the end only
        # model.fit(X_train, y_train)
        # probabilities = model.predict_proba(X_test)
        # scores['probabilities'] = probabilities[:, 1]
        # scores['y_test'] = y_test

        correct_exclusion_rate = []
        threasholds = []
        missed = []
        fscore_threashold = []
        exclusion_baseline = []
        missed_baseline = []

        # FIXME#7: Understand how to use Watanabe's split method
        # for train_index, test_index in kfold.split(X_train, y_train):

        # set up time series cross-validator
        tscv = TimeSeriesSplit(n_splits=self._n_splits)

        # FIXME#7: Understand how to use Watanabe's split method
        for train_index, test_index in tscv.split(X_train, y_train):
            X_train_index, X_test_index = X_train[train_index], X_train[test_index]
            y_train_index, y_test_index = y_train[train_index], y_train[test_index]
            model.fit(X_train_index, y_train_index)
            y_score_index = model.predict_proba(X_train_index)[:, 1]
            precision, recall, threasholds2 = metrics.precision_recall_curve(y_train_index, y_score_index)
            y_score_index = model.predict_proba(X_test_index)[:, 1]
            if (threasholds2[0] > 0.5):
                threasholds2 = [0.5]
            # matrix = metrics.confusion_matrix(y_test, [0 if i < threasholds2[0] else 1 for i in y_score])
            # correct_exclusion_rate.append(
            #     matrix[0, 0] /
            #     (matrix[0, 0] + matrix[1, 1] + matrix[0, 1] + matrix[1, 0]))
            # missed.append(matrix[1, 0] / (matrix[1, 1] + matrix[1, 0]))
            threasholds.append(threasholds2[0])
            fscore_threashold.append(metrics.f1_score(y_test_index, [0 if i < threasholds2[0] else 1 for i in y_score_index]))

            # matrix = metrics.confusion_matrix(y_test, [0 if i < 0.5 else 1 for i in y_score])
            # exclusion_baseline.append(
            #     matrix[0, 0] /
            #     (matrix[0, 0] + matrix[1, 1] + matrix[0, 1] + matrix[1, 0]))
            # missed_baseline.append(matrix[1, 0] / (matrix[1, 1] + matrix[1, 0]))

        # scores['exclusion_rate'] = correct_exclusion_rate
        scores['threasholds'] = threasholds
        # scores['missed'] = missed
        scores['fscore_threashold'] = fscore_threashold
        # scores['exclusion_baseline'] = exclusion_baseline
        # scores['missed_baseline'] = missed_baseline

        dataset['%s_scores' % self.classifier_name] = scores

        # FIXME#4: First start using the lists to storage metrics and then compute mean (until now lists are being unused)
        # # calculate the mean metrics across all folds
        # mean_accuracy = np.mean(fold_accuracies)
        # mean_precision = np.mean(fold_precisions)
        # mean_recall = np.mean(fold_recalls)
        # mean_f_measure = np.mean(fold_f_measures)

        # TODO: check if it's needed to fit the model again or not
        # fit the final model on the full training set and test it on the testing set
        # model.fit(X_train, y_train)

        # FIXME#5: Use predict_proba instead (adjust to read the right column (output will contain two columns))
        y_pred_test = model.predict(X_test)

        # compute the metrics for the test set
        accuracy_test = accuracy_score(y_test, y_pred_test)
        precision_test, recall_test, f_measure_test, _ = classification_report(y_test, y_pred_test, output_dict=True)[
            'weighted avg']

        # print the results
        # print(f"exclusion_rate: {scores['exclusion_rate']}")
        print(f"threasholds: {scores['threasholds']}")
        # print(f"missed: {scores['missed']}")
        print(f"fscore_threashold: {scores['fscore_threashold']}")
        # print(f"exclusion_baseline: {scores['exclusion_baseline']}")
        # print(f"missed_baseline: {scores['missed_baseline']}")
        print(f"Accuracy: {accuracy_test}")

        # Returns the y_test with for the currently model
        return y_pred_test, y_test

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

        params = {
            'criterion': ["gini", "entropy"],
            'max_depth': [10, 50, 100, None],
            'min_samples_split': [2, 10, 100],
            'class_weight': [None, 'balanced']
        }

        cfl = GridSearchCV(model, params, cv=5, scoring='recall')
        cfl.fit(X, y)
        for param, value in cfl.best_params_.items():
            print("%s : %s" % (param, value))
        model = tree.DecisionTreeClassifier(random_state=self._seed)
        model.set_params(**cfl.best_params_)
        return model

# Extends the SimpleClassifier class to use the SVM classifier with a specific configuration
class SVMClassifier (SimpleClassifier):
    def __init__ (self, seed=42, n_splits=3):
        SimpleClassifier.__init__(self, seed, n_splits=n_splits)
        self.classifier_name = 'svm'

    def get_classifier (self, X, y):
        print('===== SVM Classifier =====')
        print('===== Hyperparameter tunning  =====')
        params = {
            'kernel': ['linear', 'rbf'],
            'C': [1, 10, 100],
            'tol': [0.001, 0.1, 1],
            'class_weight': ['balanced', None]
        }
        model = svm.SVC(random_state=self._seed, probability=True)
        cfl = GridSearchCV(model, params, cv=5, scoring='accuracy')
        cfl.fit(X, y)
        for param, value in cfl.best_params_.items():
            print("%s : %s" % (param, value))
        model = svm.SVC(random_state=self._seed, probability=True)
        model.set_params(**cfl.best_params_)
        return model

# Author: Marcelo Costalonga

import os
import codecs
import pandas as pd
from pathlib import Path
from enum import Enum
from datetime import datetime
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from TestConfigurationLoader import TestConfiguration


# Class that analyze the results of DT and SVM by comparing with the real ones
# Writes a csv file on the "/output" dir with the format 'k<number_of_features>-report.csv' containing the results
class Report:
    _df_testing = pd.DataFrame()
    _df_training = pd.DataFrame()
    _y_test: list
    output_path = os.path.join(os.getcwd(), 'output')

    def __init__(self, training_set: dict, testing_set: dict, predictions: dict, scores: dict,
                 k_fs: int, y_true: list, result_file: str=None):
        self._df_training = pd.DataFrame.from_dict(training_set)
        self._df_testing = pd.DataFrame.from_dict(testing_set)
        self._set_result_file_path(result_file, k_fs)
        self._format_predictions(predictions)
        self._format_scores(scores)
        self._verify_output_dir()
        self._kfs = k_fs
        self._y_test = y_true

    def _verify_output_dir(self):
        # If it doesn't exist creates it
        dir_path, _ = self.result_file_path.rsplit('/', 1)
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    def _set_result_file_path(self, result_file: str, k_fs: int):
        if result_file:
            self.result_file_path = os.path.join(os.getcwd(), result_file)
        else: # If no file_path is passed, use default format
            now = datetime.now()
            month_day, hour_min = now.strftime("%b_%d,%Hh%Mm").lower().split(',')
            result_file = 'output/{}/k{}-report-{}.xlsx'.format(month_day, k_fs, hour_min)
            self.result_file_path = os.path.join(os.getcwd(), result_file)

    def _format_predictions(self, predictions: dict):
        unused_columns = ['features', 'years', 'texts']
        self._test_pred = self._df_testing.copy()
        self._test_pred.drop(unused_columns, inplace=True, axis=1)
        self._test_pred.rename(columns={'titles': 'Titles'}, inplace=True)
        self._test_pred.rename(columns={'categories': 'Was Selected?'}, inplace=True)
        self._test_proba = self._test_pred.copy()

        for clsf in predictions:
            label = clsf.upper()
            # FIXME#27: Improve this. The object 'self._df_testing' is only being used for false positives and false negatives later
            self._df_testing[label + '_pred'] = predictions[clsf]['y_pred']
            self._test_pred[label + '_pred'] = predictions[clsf]['y_pred']
            self._test_proba[label + '_proba'] = predictions[clsf]['y_proba']

    def _format_scores(self, scores: dict):
        # TODO: Improve this so we can use others classifiers without the need to specify the names of each one
        index = list()
        table = list()
        try:
            for clsf_label in scores:
                index.append(clsf_label.split('_scores')[0].upper())
                table.append(scores[clsf_label])
            self._scores = pd.DataFrame(table)
            self._scores.index = index
        except ValueError:
            # If all values are scalar will raise this error
            for clsf_label in scores:
                index.append(clsf_label.split('_scores')[0].upper())
                table.append([scores[clsf_label]])
            self._scores = pd.DataFrame(table)
            self._scores.index = index
        except Exception:
            raise Exception

    @staticmethod
    def _get_real_negatives(df: pd.DataFrame, classifier_type: str):
        return df.loc[(df['categories'] == 0) & (df['categories'] == df[classifier_type]),
                      ['titles', 'categories', classifier_type]]

    @staticmethod
    def _get_real_positives(df: pd.DataFrame, classifier_type: str):
        return df.loc[(df['categories'] == 1) & (df['categories'] == df[classifier_type]),
                      ['titles', 'categories', classifier_type]]

    @staticmethod
    def _get_false_negative(df: pd.DataFrame, classifier_type: str):
        return df.loc[(df['categories'] == 1) & (df[classifier_type] == 0), ['titles', 'categories', classifier_type]]

    @staticmethod
    def _get_false_positive(df: pd.DataFrame, classifier_type: str):
        return df.loc[(df['categories'] == 0) & (df[classifier_type] == 1), ['titles', 'categories', classifier_type]]

    @staticmethod
    def analyze_classifier(df: pd.DataFrame, y_ref: list, cls_type:str):
        df['categories'] = y_ref
        df_real_negatives = Report._get_real_negatives(df, cls_type)
        df_real_positives = Report._get_real_positives(df, cls_type)
        df_false_negative = Report._get_false_negative(df, cls_type)
        df_false_positive = Report._get_false_positive(df, cls_type)
        print("Number of Real Negatives:", len(df_real_negatives))
        print("Number of Real Positives:", len(df_real_positives))
        print("Number of False Negatives:", len(df_false_negative))
        print("Number of False Positives:", len(df_false_positive))
        all_dfs = {'REAL_NEGATIVES': df_real_negatives, 'REAL_POSITIVES': df_real_positives,
                   'FALSE_NEGATIVES': df_false_negative, 'FALSE_POSITIVES': df_false_positive
        }
        return all_dfs

    @staticmethod
    def print_detailed_results(df_dt: pd.DataFrame, df_svm: pd.DataFrame):
        for i in df_dt:
            print('\nComparing {} for both sets:'.format(i))
            result = df_dt[i].merge(df_svm[i], on='texts')
            print(result)
        return

    @staticmethod
    def compute_and_plot_ROC_AUC(y_test, y_pred, title):
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC - {}'.format(title))
        plt.legend(loc="lower right")
        plt.show(block=True)

    def report_and_write_csv(self, start: datetime, end: datetime):
        print("\nResults for Decision Tree")
        df_dt_report = Report.analyze_classifier(self._df_testing, self._y_test, 'DT_pred')
        print("\nResults for SVM")
        df_svm_report = Report.analyze_classifier(self._df_testing, self._y_test, 'SVM_pred')

        # FIXME#15: use save plot instead and organizer the output into multiple folders for each iteration
        # Report.compute_and_plot_ROC_AUC(self._y_test, self._df_testing['DT_pred'], 'DT')
        # Report.compute_and_plot_ROC_AUC(self._y_test, self._df_testing['SVM_pred'], 'SVM')

        # TODO: Comment/uncomment just for debug
        # Report.print_detailed_results(df_dt_report, df_svm_report)

        # Write more details in xlsx file
        self.create_multi_sheet_xlsx(start, end)

    @staticmethod
    def format_timedelta(start: datetime, end: datetime, fmt: str='{hours}:{minutes}:{seconds}'):
        time_diff = end - start
        time_obj = dict()
        time_obj['hours'], rem = divmod(time_diff.seconds, 3600)
        time_obj['minutes'], time_obj['seconds'] = divmod(rem, 60)
        return datetime.strptime(fmt.format(**time_obj), "%H:%M:%S").strftime("%Hh:%Mm:%Ss")

    def create_multi_sheet_xlsx(self, start: datetime, end: datetime):
        with pd.ExcelWriter(self.result_file_path) as writer:
            self._test_pred.to_excel(writer, sheet_name='Classifiers Predictions')
            self._test_proba.to_excel(writer, sheet_name='Classifiers Probabilities')
            self._scores.to_excel(writer, sheet_name='Classifiers Scores')

            # FIXME#11: Add other relevant metrics on this sheet, besides total time execution (e.g. CPU, RAM, MEMORY...)
            # Add specs information
            execution_time = Report.format_timedelta(start, end)
            specs = [
                ['Pipeline Execution Time (hours)', execution_time],
                ['Number of features', self._kfs]
            ]
            # Loading env file spec's configuration
            env_spec = TestConfiguration().get_env_vars_spec()
            specs = specs + env_spec
            df_specs = pd.DataFrame(specs, columns=['Information', 'Value'])
            df_specs.to_excel(writer, sheet_name='Test Configuration Information')

            # df_specs.to_excel(writer, sheet_name='Environment Information') # TODO: Add another sheet with Env info
            # FIXME#12.1: Add new sheet showing the names of the features selected...
            # FIXME#12.2: Add new sheet showing graphs and other statistics....
            # FIXME#12.3: Add classifiers parameters configuration

            # Add information about the classifiers' configuration
            # specs = ['Pipeline Execution Time', end - start], ['Number of features', self._kfs]
            # df_specs = pd.DataFrame(specs, columns=['Information', 'Value'])
            # df_specs.to_excel(writer, sheet_name='Test Configuration')
            #   # Dataset used - Training and Testing sizes
            #   # Parameters configuration for each classifier
            writer.save()

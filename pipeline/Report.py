# Author: Marcelo Costalonga

import os
import codecs
import pandas as pd
from pathlib import Path
from datetime import datetime

# Class that analyze the results of DT and SVM by comparing with the real ones
# Writes a csv file on the "/output" dir with the format 'k<number_of_features>-report.csv' containing the results
class Report:
    _df_testing = pd.DataFrame()
    _df_training = pd.DataFrame()
    _y_dt: list
    _y_svm: list
    output_path = os.path.join(os.getcwd(), 'output')

    def __init__(self, training_dataset: dict, testing_dataset: dict, dt_pred: list, svm_pred: list,
                 k_fs: int, y_dt: list, y_svm: list, scores: dict, result_file: str=None):
        # print(dt_pred)
        # print(svm_pred)
        self._df_training = pd.DataFrame.from_dict(training_dataset)
        self._df_testing = pd.DataFrame.from_dict(testing_dataset)
        self._set_result_file_path(result_file, k_fs)
        self._set_predictions(dt_pred, svm_pred)
        self._set_scores(scores)
        self._verify_output_dir()
        self._kfs = k_fs
        self.y_dt = y_dt
        self._y_svm = y_svm

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
            result_file = 'output/{}/k{}-report-{}.csv'.format(month_day, k_fs, hour_min)
            self.result_file_path = os.path.join(os.getcwd(), result_file)
        self.result_xlsx_file = self.result_file_path.rsplit('.', 1)[0] + '.xlsx'

    def _set_predictions(self, dt_pred: list, svm_pred: list):
        self._df_testing['DT_pred'] = dt_pred
        self._df_testing['SVM_pred'] = svm_pred

    def _set_scores(self, scores: dict):
        # TODO: Improve this so we can use others classifiers without the need to specify the names of each one
        self._dt_scores = pd.DataFrame.from_dict(scores['dt_scores'])
        self._svm_scores = pd.DataFrame.from_dict(scores['svm_scores'])

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

    def report_and_write_csv(self, start: datetime, end: datetime):
        print("\nResults for Decision Tree")
        df_dt_report = Report.analyze_classifier(self._df_testing, self.y_dt, 'DT_pred')
        print("\nResults for SVM")
        df_svm_report = Report.analyze_classifier(self._df_testing, self._y_svm, 'SVM_pred')

        # TODO: Comment/uncomment just for debug
        # Report.print_detailed_results(df_dt_report, df_svm_report)

        unused_columns = ['features', 'years', 'texts']
        self._df_testing.drop(unused_columns, inplace=True, axis=1)
        self._df_testing.rename(columns={'categories': 'Was Selected?'}, inplace=True)

        with codecs.open(self.result_file_path, 'w', encoding='utf-8') as report_file:
            self._df_testing.to_csv(report_file, index=False)

        self.create_multi_sheet_xlsx(start, end)

    def create_multi_sheet_xlsx(self, start: datetime, end: datetime):
        with pd.ExcelWriter(self.result_xlsx_file) as writer:
            self._df_testing.to_excel(writer, sheet_name='Classifiers Predictions')
            self._dt_scores.to_excel(writer, sheet_name='DT Scores')
            self._svm_scores.to_excel(writer, sheet_name='SVM Scores')

            # FIXME#11: Add other relevant metrics on this sheet, besides total time execution (e.g. CPU, RAM, MEMORY...)
            # Add specs information
            specs = ['Pipeline Execution Time', end - start], ['Number of features', self._kfs]
            df_specs = pd.DataFrame(specs, columns=['Information', 'Value'])
            df_specs.to_excel(writer, sheet_name='Environment Information')
            # FIXME#12.1: Add new sheet showing the names of the features selected...
            # FIXME#12.2: Add new sheet showing graphs and other statistics....
            # FIXME#12.3: Add classifiers parameters configuration

            # Add information about the classifiers' configuration
            specs = ['Pipeline Execution Time', end - start], ['Number of features', self._kfs]
            df_specs = pd.DataFrame(specs, columns=['Information', 'Value'])
            df_specs.to_excel(writer, sheet_name='Test Configuration')
            #   # Dataset used - Training and Testing sizes
            #   # Parameters configuration for each classifier
            writer.save()

# Author: Marcelo Costalonga

import pandas as pd
import codecs
import os

# Class that analyze the results of DT and SVM by comparing with the real ones
# Writes a csv file on the "/output" dir with the format 'k<number_of_features>-report.csv' containing the results
class Report:
    _df_testing = pd.DataFrame()
    _df_training = pd.DataFrame()
    _y_dt: list
    _y_svm: list
    output_path = os.path.join(os.getcwd(), 'output')
    def __init__(self, training_dataset: dict, testing_dataset: dict, dt_pred: list, svm_pred: list,
                 k_fs: int, y_dt: list, y_svm: list, result_file: str=None):
        print(dt_pred)
        print(svm_pred)
        # self._df_testing = pd.DataFrame.from_dict(testing_dataset)
        # self._df_training = pd.DataFrame.from_dict(training_dataset)
        if result_file:
            self._result_file_path = os.path.join(os.getcwd(), result_file)
        else:
            self._result_file_path = os.path.join(os.getcwd(), 'output/k{}-report.csv'.format(k_fs))
        self._df_testing['DT_pred'] = dt_pred
        self._df_testing['SVM_pred'] = svm_pred
        self.y_dt = y_dt
        self._y_svm = y_svm

    @staticmethod
    def get_real_negatives(df: pd.DataFrame, classifier_type: str):
        return df.loc[(df['categories'] == 0) & (df['categories'] == df[classifier_type]),
                      ['texts', 'categories', classifier_type]]

    @staticmethod
    def get_real_positives(df: pd.DataFrame, classifier_type: str):
        return df.loc[(df['categories'] == 1) & (df['categories'] == df[classifier_type]),
                      ['texts', 'categories', classifier_type]]

    @staticmethod
    def get_false_negative(df: pd.DataFrame, classifier_type: str):
        return df.loc[(df['categories'] == 1) & (df[classifier_type] == 0), ['texts', 'categories', classifier_type]]

    @staticmethod
    def get_false_positive(df: pd.DataFrame, classifier_type: str):
        return df.loc[(df['categories'] == 0) & (df[classifier_type] == 1), ['texts', 'categories', classifier_type]]

    @staticmethod
    def analyze_classifier(df: pd.DataFrame, y_ref: list, cls_type:str):
        df['categories'] = y_ref
        df_real_negatives = Report.get_real_negatives(df, cls_type)
        df_real_positives = Report.get_real_positives(df, cls_type)
        df_false_negative = Report.get_false_negative(df, cls_type)
        df_false_positive = Report.get_false_positive(df, cls_type)
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

    def report_and_write_csv(self):
        print("\nResults for Decision Tree")
        df_dt_report = Report.analyze_classifier(self._df_testing, self.y_dt, 'DT_pred')
        print("\nResults for SVM")
        df_svm_report = Report.analyze_classifier(self._df_testing, self._y_svm, 'SVM_pred')

        # TODO: Comment/uncomment just for debug
        # Report.print_detailed_results(df_dt_report, df_svm_report)

        unused_columns = ['features', 'years']
        self._df_testing.drop(unused_columns, inplace=True, axis=1)
        self._df_testing.rename(columns={'categories': 'Was Selected?'}, inplace=True)
        with codecs.open(self._result_file_path, 'w', encoding='utf-8') as report_file:
            self._df_testing.to_csv(report_file, index=False)

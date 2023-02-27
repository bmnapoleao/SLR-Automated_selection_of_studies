# Author: Marcelo Costalonga

import pandas as pd
import codecs
import os

# Class that analyze the results of DT and SVM by comparing with the real ones
# Writes a csv file on the "/output" dir with the format 'k<number_of_features>-report.csv' containing the results
class Report:
    output_path = os.path.join(os.getcwd(), 'output')
    def __init__(self, testing_dataset: dict, training_dataset: dict, dt_pred: list, svm_pred: list, k_fs: int):
        self._df_testing = pd.DataFrame.from_dict(testing_dataset)
        self._df_training = pd.DataFrame.from_dict(training_dataset)
        self._resuls_file_path = os.path.join(os.getcwd(), 'output/k{}-report.csv'.format(k_fs))
        self._df_testing['DT_pred'] = dt_pred
        self._df_testing['SVM_pred'] = svm_pred

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
    def analyze_classifier(df: pd.DataFrame, cls_type:str):
        df_real_negatives = Report.get_real_negatives(df, cls_type)
        df_real_positives = Report.get_real_positives(df, cls_type)
        df_false_negative = Report.get_false_negative(df, cls_type)
        df_false_positive = Report.get_false_positive(df, cls_type)
        print("Number of Real Negatives:", len(df_real_negatives))
        print("Number of Real Positives:", len(df_real_positives))
        print("Number of False Negatives:", len(df_false_negative))
        print("Number of False Positives:", len(df_false_positive))

    def report_and_write_csv(self):
        print("\nResults for Decision Tree")
        Report.analyze_classifier(self._df_testing, 'DT_pred')
        print("\nResults for SVM")
        Report.analyze_classifier(self._df_testing, 'SVM_pred')

        unused_columns = ['features', 'years']
        self._df_testing.drop(unused_columns, inplace=True, axis=1)
        self._df_testing.rename(columns={'categories': 'Was Selected?'}, inplace=True)
        with codecs.open(self._resuls_file_path, 'w', encoding='utf-8') as report_file:
            self._df_testing.to_csv(report_file, index=False)
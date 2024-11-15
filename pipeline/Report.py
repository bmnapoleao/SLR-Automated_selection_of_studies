# Author: Marcelo Costalonga

import os
import pandas as pd
from pathlib import Path
from datetime import datetime
from TestConfigurationLoader import TestConfiguration


# Class that writes a report file with tge details of the execution
class Report:
    _classifiers_labels = list()
    _y_test: list

    def __init__(self, training_set: dict, testing_set: dict, clsf_exec_results: dict,
                 k_fs: int, y_true: list, start_time: datetime, end_time: datetime, result_file: str=None):
        self._df_specs = None
        self._df_scores = None
        self._df_analysis = None
        self._kfs = k_fs
        self._y_test = y_true
        self._start = start_time
        self._end = end_time
        self.result_file_path = os.path.join(os.getcwd(), result_file)
        self._verify_output_dir()
        self._set_classifiers_labels(clsf_exec_results)
        self._set_predictions(testing_set, clsf_exec_results)
        self._set_scores(clsf_exec_results)
        self._set_best_params(clsf_exec_results)

    def _verify_output_dir(self):
        # If output directory doesn't exist, creates it
        dir_path, _ = self.result_file_path.rsplit('/', 1)
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    def _set_classifiers_labels(self, clsf_report: dict):
        try:
            self._classifiers_labels = [label.upper() for label in clsf_report]
        except Exception:
            raise Exception

    def _set_predictions(self, testing_set: dict, clsf_exec_results: dict):
        unused_columns = ['features', 'years', 'texts']
        self._df_test_pred = pd.DataFrame.from_dict(testing_set)
        self._df_test_pred.drop(unused_columns, inplace=True, axis=1)
        self._df_test_pred.rename(columns={'titles': 'Titles'}, inplace=True)
        self._df_test_pred.rename(columns={'categories': 'Was Selected?'}, inplace=True)
        self._df_test_proba = self._df_test_pred.copy()
        for clsf in clsf_exec_results:
            label = clsf.upper()
            self._df_test_pred[label + '_pred'] = clsf_exec_results[clsf]['predictions']['y_pred']
            self._df_test_proba[label + '_proba'] = clsf_exec_results[clsf]['predictions']['y_proba']


    def _set_scores(self, clsf_exec_results: dict):
        index = self._classifiers_labels
        table = list()
        try:
            for clsf_label in clsf_exec_results:
                table.append(clsf_exec_results[clsf_label]['scores'])
            self._df_scores = pd.DataFrame(table)
            self._df_scores.index = index
        except ValueError:
            # If all values are scalar will raise this error
            for clsf_label in clsf_exec_results:
                table.append([clsf_exec_results[clsf_label]['scores']])
            self._df_scores = pd.DataFrame(table)
            self._df_scores.index = index
        except Exception:
            raise Exception

    def _set_best_params(self, clsf_exec_results: dict):
        index = list()
        for label in self._classifiers_labels:
            index.append('{}: Best Params'.format(label))
            index.append('{}: Tested Params'.format(label))
        table = list()
        try:
            for clsf_label in clsf_exec_results:
                table.append(clsf_exec_results[clsf_label]['best_params'])
                table.append(clsf_exec_results[clsf_label]['tested_params'])
            self._df_params = pd.DataFrame(table)
            self._df_params.index = index
        except ValueError:
            # If all values are scalar will raise this error
            for clsf_label in clsf_exec_results:
                table.append([clsf_exec_results[clsf_label]['best_params']])
                table.append([clsf_exec_results[clsf_label]['tested_params']])
            self._df_params = pd.DataFrame(table)
            self._df_params.index = index
        except Exception:
            raise Exception

    def _format_analysis(self):
        index = self._classifiers_labels
        table = list()
        try:
            for clsf_label in self._classifiers_labels:
                table.append(Report.analyze_classifier(self._df_test_pred, clsf_label, 'Was Selected?'))
            self._df_analysis = pd.DataFrame(table)
            self._df_analysis.index = index
        except ValueError:
            # If all values are scalar will raise this error
            for clsf_label in self._classifiers_labels:
                table.append([Report.analyze_classifier(self._df_test_pred, clsf_label, 'Was Selected?')])
            self._df_analysis = pd.DataFrame(table)
            self._df_analysis.index = index
        except Exception:
            raise Exception

    @staticmethod
    def format_report_file_path(output_path, k_features):
        if output_path:
            if output_path.endswith('.xlsx') or output_path.endswith('.csv'):
                # If output path contains output_dir/output_file_name
                report_file_name = output_path.split('/')[-1]
                output_path = output_path.rstrip(report_file_name)
                report_file_name = report_file_name.replace('.csv', '.xlsx')
            else:
                # If output path contains only output_dir, use default output_file_name format
                now = datetime.now()
                month_day, hour_min = now.strftime("%b%d,%Hh%Mm").lower().split(',')
                report_file_name = 'k{}-report-{}-{}.xlsx'.format(k_features, month_day, hour_min)
        else:
            print('ERROR: No output path found')
            raise Exception

        report_file_path = os.path.join(output_path, report_file_name)
        if os.path.exists(report_file_path):
            # Assert the file doesn't exist already
            print('[Error] File "{}" already exists, please inform a new file path for output'.format(
                report_file_path))
            raise FileExistsError
        return report_file_path

    @staticmethod
    def _get_true_negatives(df: pd.DataFrame, classifier_type: str, y_key: str='categories'):
        return df.loc[(df[y_key] == 0) & (df[y_key] == df[classifier_type]),
                      ['Titles', y_key, classifier_type]]

    @staticmethod
    def _get_true_positives(df: pd.DataFrame, classifier_type: str, y_key: str='categories'):
        return df.loc[(df[y_key] == 1) & (df[y_key] == df[classifier_type]),
                      ['Titles', y_key, classifier_type]]

    @staticmethod
    def _get_false_negative(df: pd.DataFrame, classifier_type: str, y_key: str='categories'):
        return df.loc[(df[y_key] == 1) & (df[classifier_type] == 0), ['Titles', y_key, classifier_type]]

    @staticmethod
    def _get_false_positive(df: pd.DataFrame, classifier_type: str, y_key: str='categories'):
        return df.loc[(df[y_key] == 0) & (df[classifier_type] == 1), ['Titles', y_key, classifier_type]]

    @staticmethod
    def analyze_classifier(df: pd.DataFrame, cls_type: str, y_key: str='categories'):
        y_pred_key = cls_type + '_pred'
        # For now, we are only using to count how many were Ture/False Pos/Neg, but to see which title was predicted
        # right or wrong just check the Dataframe returned by each method instead of its length
        true_negatives_num = len(Report._get_true_negatives(df, y_pred_key, y_key))
        true_positives_num = len(Report._get_true_positives(df, y_pred_key, y_key))
        false_negative_num = len(Report._get_false_negative(df, y_pred_key, y_key))
        false_positive_num = len(Report._get_false_positive(df, y_pred_key, y_key))
        print("\nResults for {}".format(cls_type))
        print("Number of True Negatives:", true_negatives_num)
        print("Number of True Positives:", true_positives_num)
        print("Number of False Negatives:", false_negative_num)
        print("Number of False Positives:", false_positive_num)
        return {'TRUE_NEGATIVES': true_negatives_num, 'TRUE_POSITIVES': true_positives_num,
                'FALSE_NEGATIVES': false_negative_num, 'FALSE_POSITIVES': false_positive_num}

    @staticmethod
    def print_detailed_results(df_dt: pd.DataFrame, df_svm: pd.DataFrame):
        for i in df_dt:
            print('\nComparing {} for both sets:'.format(i))
            result = df_dt[i].merge(df_svm[i], on='texts')
            print(result)
        return

    @staticmethod
    # Formats the difference between two datetime objects to look like "%Hh:%Mm:%Ss" (e.g. "00h:00m:00s")
    def format_timedelta(start: datetime, end: datetime, fmt: str='{hours}:{minutes}:{seconds}'):
        time_diff = end - start
        time_obj = dict()
        time_obj['hours'], rem = divmod(time_diff.seconds, 3600)
        time_obj['minutes'], time_obj['seconds'] = divmod(rem, 60)
        return datetime.strptime(fmt.format(**time_obj), "%H:%M:%S").strftime("%Hh:%Mm:%Ss")

    def _format_specs(self):
        # Add specs information
        execution_time = Report.format_timedelta(self._start, self._end)
        specs = [
            ['Pipeline Execution Time (hours)', execution_time],
            ['Number of features', self._kfs]
        ]
        # Loading env file spec's configuration
        env_spec = TestConfiguration().get_env_vars_spec()
        specs = specs + env_spec
        self._df_specs = pd.DataFrame(specs, columns=['Information', 'Value'])

    # Writes a xlsx file with multiple sheets, each sheet contains a different information about the execution
    def report_and_write_csv(self):
        self._format_specs()
        self._format_analysis()

        with pd.ExcelWriter(self.result_file_path) as writer:
            self._df_scores.to_excel(writer, sheet_name='Scores')
            self._df_analysis.to_excel(writer, sheet_name='Analysis')
            self._df_test_pred.to_excel(writer, sheet_name='Predictions')
            self._df_test_proba.to_excel(writer, sheet_name='Probabilities')
            self._df_params.to_excel(writer, sheet_name='GridSearch Parameters')
            self._df_specs.to_excel(writer, sheet_name='Test Configuration')
            writer.save()



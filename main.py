# Author: Marcelo Costalonga
import os.path

# from pipeline.Classifiers import DecisionTreeClassifier, SVMClassifier, KNNClassifier, RandomForestClassifier
# from pipeline.Classifiers import GaussianNaiveBayesClassifier, LinearRegressionClassifier, LogisticRegressionClassifier
from pipeline.Classifiers import SVMClassifier, RandomForestClassifier
from pipeline.DatasetGenerator import DatasetGenerator
from pipeline.FeaturesSelector import FeaturesSelector
from pipeline.InputValidator import InputValidator
from pipeline.Report import Report
from pipeline.TextFiltering import TextFilter
from datetime import datetime
from TestConfigurationLoader import TestConfiguration
import sys

DEFAULT_OUTPUT_DIR = 'reports/'

if __name__ == '__main__':
    start = datetime.now()

    # Assert user passed required parameters
    output_path = None
    try:
        assert len(sys.argv) > 1
        k_features = int(sys.argv[1])
        try:
            assert 0 < k_features <= 5000
        except Exception:
            print("\nMissing parameter with the number of kFeatures. Please inform in the command line.")
            raise Exception

        try:
            env_file_path = sys.argv[2]
            assert env_file_path.endswith('.env')
        except Exception as e:
            print(e)
            raise

    except AssertionError:
        exit(0)

    # Start Pipeline
    # Loading test configuration variables
    try:
        test_config = TestConfiguration(file_path=env_file_path)
    except Exception:
        print("\nProblem while trying to load the test env file")
        raise Exception


    try: # Output path, can be: dir/file, dir/ or none
        output_path = sys.argv[3]
    except IndexError:
        output_sufix = 'scoring_{}'.format(test_config.get_grid_search_params()['gs_scoring'])
        output_path = DEFAULT_OUTPUT_DIR + env_file_path.split('.')[0].split('/')[-1]
        output_path = os.path.join(output_path, output_sufix)
        print("\n[WARNING:  NO OUTPUT PATH INFORMED] Using default path. "
              "The result file will be created at {}.\n".format(output_path))

    report_file_path = Report.format_report_file_path(output_path, k_features)
    print('REPORT FILE:', report_file_path)

    # Validates input files
    validator = InputValidator()
    validator.execute()

    # Apply text filtering techniques
    text_filter = TextFilter(training_set=validator.training_set, testing_set=validator.testing_set)
    text_filter.execute()

    # Converts datasets and apply text vectorization to extract features
    dataset_generator = DatasetGenerator(filtered_training_set=text_filter.filtered_training_set,
                                         filtered_testing_set=text_filter.filtered_testing_set)
    dataset_generator.execute()

    # Select k best features
    feature_selector = FeaturesSelector(k_fs=k_features)
    feature_selector.execute(training_dataset=dataset_generator.training_dataset,
                             testing_dataset=dataset_generator.testing_dataset)

    # Perform ML test
    number_of_splits = 5
    clsf_exec_results = dict()

    if test_config.use_feature_selection():
        # # USING FEATURE SELECTION
        training_set = feature_selector.training_dataset
        testing_set = feature_selector.testing_dataset
    else:
        # # WITHOUT FEATURE SELECTION (TF_IDF only)
        training_set = dataset_generator.training_dataset
        testing_set = dataset_generator.testing_dataset

    # # DT
    # dt_classifier = DecisionTreeClassifier(seed=42, n_splits=number_of_splits)
    # clsf_exec_results['dt'] = dt_classifier.execute(training_set=training_set, testing_set=testing_set)

    # SVM
    svm_classifier = SVMClassifier(seed=42, n_splits=number_of_splits)
    clsf_exec_results['svm'] = svm_classifier.execute(training_set=training_set, testing_set=testing_set)

    # Random Forest
    rf_classifier = RandomForestClassifier(seed=42, n_splits=number_of_splits)
    clsf_exec_results['rforest'] = rf_classifier.execute(training_set=training_set, testing_set=testing_set)
    
    end = datetime.now()

    # Compare results and generate reports
    report = Report(training_set=training_set, testing_set=testing_set, clsf_exec_results=clsf_exec_results,
                    k_fs=k_features, y_true=testing_set['categories'], start_time=start,
                    end_time=end, result_file=report_file_path)
    report.report_and_write_csv()
    print("\nExecution ended:\n\tStart: {}\n\tEnd: {}\n\tTotal time duration = {}".format(start, end, end-start))
    print("Results can be seem at:\n\t - {}".format(report.result_file_path))
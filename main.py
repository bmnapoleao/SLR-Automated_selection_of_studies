# Author: Marcelo Costalonga
import os.path

from pipeline.Classifiers import DecisionTreeClassifier, SVMClassifier, KNNClassifier, RandomForestClassifier
from pipeline.Classifiers import GaussianNaiveBayesClassifier, LinearRegressionClassifier, LogisticRegressionClassifier
from pipeline.DatasetGenerator import DatasetGenerator
from pipeline.FeaturesSelector import FeaturesSelector
from pipeline.InputValidator import InputValidator
from pipeline.Report import Report
from pipeline.TextFiltering import TextFilter
from pipeline.BibFormater import BibFormater
from datetime import datetime
from TestConfigurationLoader import TestConfiguration
import sys


if __name__ == '__main__':

    # TODO: OBS: Para executar multiplas vezes pelo terminal:
    # # for i in {1..5}; do python main.py "$((i*1000))"; done # (variando k de 1000 a 5000)
    # # for i in {0..4}; do python main.py "$((i*1000+500))"; done # (variando k de 500 a 4500)
    # FIXME #2: Check integrity of dataset Title: "PMBOK Guides" seems to be a Book with an abstract field that doens't looks valid

    # TODO: Delete this or comment when done testing
    seed = None
    start = datetime.now()
    # TODO: Change to save the results in a specific path/file
    month_day, hour_min = start.strftime("%b%d,%Hh%Mm").lower().split(',')
    # result_file = 'output/small-samples-{}/k{}-report-{}.csv'.format(month_day, k_features, hour_min)
    # result_file = 'output/tests-with-inverted-dataset-{}/k{}-report-{}.csv'.format(month_day, k_features, hour_min)
    # result_file = None


    # Hardcoded params (just for testing on ide)
    k_features = 1000 # TODO: max is around 5k
    # env_file_path = '/home/mcostalonga/new-home/thesis/git-repo-slr/SLR-Automated_selection_of_studies/test_config_env_files/dummy-test-file.env'

    # # # Assert user passed required parameters
    output_path = None
    try:
        assert len(sys.argv) > 1
        k_features = int(sys.argv[1])
        env_file_path = sys.argv[2]  # TODO: add check to assert env file exists
        try:
            assert 0 < k_features <= 5000
        except Exception:
            raise Exception
        try:
            output_path = sys.argv[3]
        except Exception:
            pass
    except AssertionError:
        print("\nMissing parameter with the number of kFeatures. Please inform in the command line.")
        exit(0)

    report_file_name = 'k{}-report-{}-{}.xlsx'.format(k_features, month_day, hour_min)
    if not output_path:
        output_path = 'output-april/recall-tests-with-TFIDF'
    elif output_path.endswith('.xlsx') or output_path.endswith('.csv'):
        report_file_name = output_path.split('/')[-1]
        output_path = output_path.rstrip(report_file_name)
        report_file_name = report_file_name.replace('.csv', '.xlsx')

    report_file_path = os.path.join(output_path, report_file_name)
    print('REPORT FILE:', report_file_path)
    # result_file = 'output-april/recall-tests-with-TFIDF/TMP-k{}-report-{}-{}.csv'.format(k_features,month_day,hour_min)

    #### END OF CONFIGURATION / BEGIN OF PIPELINE
    #### FIXME#21: Organize code

    # Loading test configuration variables
    try:
        test_config = TestConfiguration(file_path=env_file_path)
    except Exception:
        print("\nProblem while trying to load the test env file")
        raise Exception

    # Validates input files
    validator = InputValidator()
    validator.execute()

    # TODO: Uncomment with small sets
    # print(validator.testing_set)
    # print(validator.training_set)

    # Print number of studies by year of each set
    BibFormater.get_dataset_years(validator.training_set, validator.testing_set)

    # Apply text filtering techniques
    text_filter = TextFilter(training_set=validator.training_set, testing_set=validator.testing_set)
    text_filter.execute()

    # Converts datasets and apply text vectorization to extract features
    # dataset_generator = DatasetGenerator(filtered_dataset=text_filter.filtered_dataset) # TODO#: Single dataset vs Two datasets
    dataset_generator = DatasetGenerator(filtered_training_set=text_filter.filtered_training_set,
                                         filtered_testing_set=text_filter.filtered_testing_set)
    dataset_generator.execute()

    # Select k best features
    feature_selector = FeaturesSelector(k_fs=k_features)
    # dataset_fs = feature_selector.execute(dataset_generator.dataset) # TODO#: Single dataset vs Two datasets
    feature_selector.execute(training_dataset=dataset_generator.training_dataset,
                             testing_dataset=dataset_generator.testing_dataset)

    # Perform ML test
    number_of_splits = 5 # TODO: originally set to 3
    scores_report = dict()
    predictions_report = dict()

    if test_config.use_feature_selection():
        # # USING FEATURE SELECTION
        training_set = feature_selector.training_dataset
        testing_set = feature_selector.testing_dataset
    else:
        # # WITHOUT FEATURE SELECTION (TF_IDF only)
        training_set = dataset_generator.training_dataset
        testing_set = dataset_generator.testing_dataset

    # # Decision Tree
    dt_classifier = DecisionTreeClassifier(seed=42, criterion='gini', n_splits=number_of_splits)
    predictions_report['dt'], scores_report['dt_scores'] = dt_classifier.execute(training_set=training_set,
                                                                                 testing_set=testing_set)
    # # SVM
    svm_classifier = SVMClassifier(seed=42, n_splits=number_of_splits)
    predictions_report['svm'], scores_report['svm_scores'] = svm_classifier.execute(training_set=training_set,
                                                                                    testing_set=testing_set)

    # # Random Forest
    rf_classifier = RandomForestClassifier(seed=42, n_splits=number_of_splits)
    predictions_report['rforest'], scores_report['rforest_scores'] = rf_classifier.execute(training_set=training_set,
                                                                                           testing_set=testing_set)

    # # KNN
    knn_classifier = KNNClassifier(seed=42, n_splits=number_of_splits)
    predictions_report['knn'], scores_report['knn_scores'] = knn_classifier.execute(training_set=training_set,
                                                                                    testing_set=testing_set)

    # # # Naive Bayes (Gaussian)
    # gnb_classifier = GaussianNaiveBayesClassifier(seed=42, n_splits=number_of_splits)
    # predictions_report['gnb'], scores_report['gnb_scores'] = gnb_classifier.execute(training_set=training_set,
    #                                                                                 testing_set=testing_set)
    #
    # # # Linear Regression
    # linr_classifier = LinearRegressionClassifier(seed=42)
    # predictions_report['linr'], scores_report['linr_scores'] = linr_classifier.execute(training_set=training_set,
    #                                                                                    testing_set=testing_set)
    #
    # # # Logistic Regression
    # logr_classifier = LogisticRegressionClassifier(seed=42)
    # predictions_report['logr'], scores_report['logr_scores'] = knn_classifier.execute(training_set=training_set,
    #                                                                                   testing_set=testing_set)

    # Compare results and generate reports
    report = Report(training_set=training_set, testing_set=testing_set,
                    predictions=predictions_report, scores=scores_report,
                    # dt_pred=predictions_report['dt'], svm_pred=predictions_report['svm'],
                    k_fs=k_features, y_true=testing_set['categories'], result_file=report_file_path)
    end = datetime.now()
    report.report_and_write_csv(start, end)
    print("\nExecution ended:\n\tStart: {}\n\tEnd: {}\n\tTotal time duration = {}".format(start, end, end-start))
    print("Results can be seem at:\n\t - {}".format(report.result_file_path))

# TODO: for i in {1,2,4}; do python main.py "$((i*5000))"; done (OBS tot features = 121279 rethink about kFs range)
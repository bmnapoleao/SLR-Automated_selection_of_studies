# Author: Marcelo Costalonga

from pipeline.Classifiers import DecisionTreeClassifier, SVMClassifier
from pipeline.DatasetGenerator import DatasetGenerator
from pipeline.FeaturesSelector import FeaturesSelector
from pipeline.InputValidator import InputValidator
from pipeline.Report import Report
from pipeline.TextFiltering import TextFilter
from pipeline.BibFormater import BibFormater
from datetime import datetime

if __name__ == '__main__':


    # # Assert user passed required parameters
    # try:
    #     assert len(sys.argv) > 1
    # except AssertionError:
    #     print("\nMissing parameter with the number of kFeatures. Please inform in the command line.")
    #     exit(0)
    # number_of_features = int(sys.argv[1])
    number_of_features = 5000 # TODO: max is around 5k

    # FIXME #2: Check integrity of dataset Title: "PMBOK Guides" seems to be a Book with an abstract field that doens't looks valid

    # TODO: Delete this or comment when done testing
    seed = None
    start = datetime.now()
    # TODO: Change to save the results in a specific path/file
    # month_day, hour_min = start.strftime("%b_%d,%Hh%Mm").lower().split(',')
    # result_file = 'output/small-dataset-{}/k{}-report-{}.csv'.format(month_day, number_of_features, hour_min)
    result_file = None

    # Validates input files
    validator = InputValidator()
    validator.execute()
    try:
        assert validator.is_valid() == True
    except AssertionError:
        print("\nPlease correct the errors found on the bib files before executing the program again.")
        exit(0)

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
    feature_selector = FeaturesSelector(k_fs=number_of_features)
    # dataset_fs = feature_selector.execute(dataset_generator.dataset)
    feature_selector.execute(training_dataset=dataset_generator.training_dataset,
                             testing_dataset=dataset_generator.testing_dataset)

    # Perform ML test
    number_of_splits = 3 # TODO: originally set to 3
    scores_report = dict()

    # # Decision Tree
    dt_classifier = DecisionTreeClassifier(seed=42, criterion='gini', n_splits=number_of_splits)
    dt_predictions, y_ref_dt, dt_scores = dt_classifier.execute(training_dataset=feature_selector.training_dataset,
                                                                testing_dataset=feature_selector.testing_dataset)
    scores_report['dt_scores'] = dt_scores

    # # SVM
    svm_classifier = SVMClassifier(seed=42, n_splits=number_of_splits)
    svm_predictions, y_ref_svm, svm_scores = svm_classifier.execute(training_dataset=feature_selector.training_dataset,
                                                                    testing_dataset=feature_selector.testing_dataset)
    scores_report['svm_scores'] = svm_scores

    # Compare results and generate reports
    report = Report(training_dataset=feature_selector.training_dataset,
                    testing_dataset=feature_selector.testing_dataset,
                    dt_pred=dt_predictions, svm_pred=svm_predictions, k_fs=number_of_features,
                    y_dt=y_ref_dt, y_svm=y_ref_svm, scores=scores_report, result_file=result_file)
    end = datetime.now()
    report.report_and_write_csv(start, end)
    print("\nExecution ended:\n\tStart: {}\n\tEnd: {}\n\tTotal time duration = {}".format(start, end, end-start))
    print("Results can be seem at:\n\t - {} (simple csv with classifiers predictions) \n\t - {} (complete information "
          "about each classifiers, machine specs and etc)".format(report.result_file_path, report.result_xlsx_file))


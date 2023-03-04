# Author: Marcelo Costalonga

from pipeline.Classifiers import DecisionTreeClassifier, SVMClassifier
from pipeline.DatasetGenerator import DatasetGenerator
from pipeline.FeaturesSelector import FeaturesSelector
from pipeline.InputValidator import InputValidator
from pipeline.Report import Report
from pipeline.TextFiltering import TextFilter
import sys

from pipeline_v2.BibFormater import BibFormater


if __name__ == '__main__':

    # # Assert user passed required parameters
    # try:
    #     assert len(sys.argv) > 1
    # except AssertionError:
    #     print("\nMissing parameter with the number of kFeatures. Please inform in the command line.")
    #     exit(0)
    # number_of_features = int(sys.argv[1])
    number_of_features = 50

    # TODO: Delete this or comment when done testing
    seed = None
    result_file = 'output/ourDataSet-Inverted-k{}-report-CLSv1.csv'.format(seed, number_of_features)

    # Validates input files
    validator = InputValidator()
    validator.execute()
    try:
        assert validator.is_valid() == True
    except AssertionError:
        print("\nPlease correct the errors found on the bib files before executing the program again.")
        exit(0)

    # Apply text filtering techniques
    text_filter = TextFilter(testing_set=validator.testing_set, training_set=validator.training_set)
    text_filter.execute()

    # TODO: TESTING (DELETE AFTER - 27/02/23)
    print('\n\t TRAINING SET YEARS:')
    BibFormater.get_dataset_years(text_filter.filtered_training_set)
    print('\n\t TESTING SET YEARS:')
    BibFormater.get_dataset_years(text_filter.filtered_testing_set)
    # TODO: TESTING (DELETE AFTER - 27/02/23)


    # Converts datasets and apply text vectorization to extract features
    dataset_generator = DatasetGenerator(filtered_testing_set=text_filter.filtered_testing_set,
                                         filtered_training_set=text_filter.filtered_training_set)
    dataset_generator.execute()

    # Select k best features
    feature_selector = FeaturesSelector(k_fs=number_of_features)
    training_dataset_fs = feature_selector.execute(dataset_generator.training_dataset)
    testing_dataset_fs = feature_selector.execute(dataset_generator.testing_dataset)

    # Perform ML test
    number_of_splits = 5 # TODO: originally set to 3

    # # Decision Tree
    dt_classifier = DecisionTreeClassifier(seed=42, criterion='gini', n_splits=number_of_splits)
    dt_predictions = dt_classifier.execute(training_dataset_fs, testing_dataset_fs)

    # # SVM
    svm_classifier = SVMClassifier(seed=42, n_splits=number_of_splits)
    svm_predictions = svm_classifier.execute(training_dataset_fs, testing_dataset_fs)

    # Compare results and generate reports
    report = Report(training_dataset=training_dataset_fs, testing_dataset=testing_dataset_fs,
                    dt_pred=dt_predictions, svm_pred=svm_predictions, k_fs=number_of_features)
    report.report_and_write_csv()


# ### TODO: Test 2nd approach
# # from pipeline_v2.Classifiers import DecisionTreeClassifier, SVMClassifier
# # from pipeline_v2.DatasetGenerator import DatasetGenerator
# # from pipeline_v2.FeaturesSelector import FeaturesSelector
# # from pipeline_v2.Report import Report
# # from pipeline_v2.TextFiltering import TextFilter
# # import sys
# from pipeline_v2.BibFormater import BibFormater
#
# if __name__ == '__main__':
#
#     # # Assert user passed required parameters
#     # try:
#     #     assert len(sys.argv) > 1
#     # except AssertionError:
#     #     print("\nMissing parameter with the number of kFeatures. Please inform in the command line.")
#     #     exit(0)
#     # number_of_features = int(sys.argv[1])
#
#     number_of_features = 100
#     seed = 5
#
#     # Validates input files
#     validator = BibFormater(seed, number_of_features, verbose=False)
#     validator.execute()
#
#     # Apply text filtering techniques
#     text_filter = TextFilter(testing_set=validator.testing_set, training_set=validator.training_set)
#     text_filter.execute()
#
#     # Converts datasets and apply text vectorization to extract features
#     dataset_generator = DatasetGenerator(filtered_testing_set=text_filter.filtered_testing_set,
#                                          filtered_training_set=text_filter.filtered_training_set)
#     dataset_generator.execute()
#
#     # Select k best features
#     feature_selector = FeaturesSelector(k_fs=number_of_features)
#     training_dataset_fs = feature_selector.execute(dataset_generator.training_dataset)
#     testing_dataset_fs = feature_selector.execute(dataset_generator.testing_dataset)
#
#     # Perform ML test
#     number_of_splits = 3 # TODO: originally set to 3
#
#     # # Decision Tree
#     dt_classifier = DecisionTreeClassifier(seed=42, criterion='gini', n_splits=number_of_splits)
#     dt_predictions = dt_classifier.execute(training_dataset_fs, testing_dataset_fs)
#
#     # # SVM
#     svm_classifier = SVMClassifier(seed=42, n_splits=number_of_splits)
#     svm_predictions = svm_classifier.execute(training_dataset_fs, testing_dataset_fs)
#
#     # Compare results and generate reports
#     result_file = 'output/tmp-seed{}-k{}-report-CLSv1.csv'.format(seed, number_of_features)
#     report = Report(training_dataset=training_dataset_fs, testing_dataset=testing_dataset_fs,
#                     dt_pred=dt_predictions, svm_pred=svm_predictions, k_fs=number_of_features, result_file=result_file)
#     report.report_and_write_csv()


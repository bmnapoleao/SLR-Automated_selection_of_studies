# Author: Marcelo Costalonga

from pipeline.Classifiers import DecisionTreeClassifier, SVMClassifier
from pipeline.DatasetGenerator import DatasetGenerator
from pipeline.FeaturesSelector import FeaturesSelector
from pipeline.InputValidator import InputValidator
from pipeline.Report import Report
from pipeline.TextFiltering import TextFilter
from sklearn.feature_selection import SelectKBest

from pipeline.BibFormater import BibFormater


if __name__ == '__main__':

    # # Assert user passed required parameters
    # try:
    #     assert len(sys.argv) > 1
    # except AssertionError:
    #     print("\nMissing parameter with the number of kFeatures. Please inform in the command line.")
    #     exit(0)
    # number_of_features = int(sys.argv[1])
    number_of_features = 1500

    # TODO: Delete this or comment when done testing
    seed = None
    result_file = 'output/new-samples-CLSv4-k{}-report.csv'.format(number_of_features)

    # Validates input files
    validator = InputValidator()
    validator.execute()
    try:
        assert validator.is_valid() == True
    except AssertionError:
        print("\nPlease correct the errors found on the bib files before executing the program again.")
        exit(0)

    print(validator.testing_set)
    print(validator.training_set)

    # Apply text filtering techniques
    text_filter = TextFilter(training_set=validator.training_set, testing_set=validator.testing_set)
    text_filter.execute()

    # TODO: TESTING (DELETE AFTER - 27/02/23)
    print('\n\t TRAINING SET YEARS:')
    BibFormater.get_dataset_years(validator.training_set)
    print('\n\t TESTING SET YEARS:')
    BibFormater.get_dataset_years(validator.testing_set)
    # TODO: TESTING (DELETE AFTER - 27/02/23)


    # Converts datasets and apply text vectorization to extract features
    dataset_generator = DatasetGenerator(filtered_dataset=text_filter.filtered_dataset)
    dataset_generator.execute()

    # Select k best features
    feature_selector = FeaturesSelector(k_fs=number_of_features)
    # dataset_fs = feature_selector.execute(dataset_generator.dataset)
    fs = feature_selector.execute(dataset_generator.dataset)

    # Perform ML test
    number_of_splits = 5 # TODO: originally set to 3

    # # Decision Tree
    dt_classifier = DecisionTreeClassifier(seed=42, criterion='gini', n_splits=number_of_splits)
    dt_predictions = dt_classifier.execute(df_training_set=feature_selector.df_training_set,
                                           df_testing_set=feature_selector.df_testing_set, fs=fs)

    # # SVM
    svm_classifier = SVMClassifier(seed=42, n_splits=number_of_splits)
    svm_predictions = svm_classifier.execute(df_training_set=feature_selector.df_training_set,
                                             df_testing_set=feature_selector.df_testing_set, fs=fs)

    # Compare results and generate reports
    report = Report(training_dataset=feature_selector.df_training_set, testing_dataset=feature_selector.df_testing_set,
                    dt_pred=dt_predictions, svm_pred=svm_predictions, k_fs=number_of_features)
    report.report_and_write_csv()

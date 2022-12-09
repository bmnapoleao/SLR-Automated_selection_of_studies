# Author: Marcelo Costalonga

from pipeline.DatasetGenerator import DatasetGenerator
from pipeline.FeaturesSelector import FeaturesSelector
from pipeline.InputValidator import BibValidator
from pipeline.TextFiltering import TextFilter
import sys

if __name__ == '__main__':

    # Assert user passed required parameters
    try:
        assert len(sys.argv) > 1
    except AssertionError:
        print("\nMissing parameter with the number of kFeatures. Please inform in the command line.")
        exit(0)
    number_of_features = int(sys.argv[1])

    # Validates input files
    validator = BibValidator()
    validator.execute()
    try:
        assert validator.is_valid() == True
    except AssertionError:
        print("\nPlease correct the errors found on the bib files before executing the program again.")
        exit(0)

    # Apply text filtering techniques
    text_filter = TextFilter(testing_set=validator.testing_set, training_set=validator.training_set)
    text_filter.execute()

    # Converts datasets and apply text vectorization to extract features
    dataset_generator = DatasetGenerator(filtered_testing_set=text_filter.filtered_testing_set,
                                         filtered_training_set=text_filter.filtered_training_set)
    dataset_generator.execute()

    # Select k best features
    feature_selector = FeaturesSelector(k_fs=number_of_features)
    testing_dataset_fs = feature_selector.execute(dataset_generator.testing_dataset)
    training_dataset_fs = feature_selector.execute(dataset_generator.training_dataset)


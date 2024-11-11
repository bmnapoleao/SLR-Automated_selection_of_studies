# Author: Marcelo Costalonga
# (code adapted from: https://github.com/watinha/automatic-selection-slr/blob/master/pipeline/__init__.py)

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Class that converts lists of dicts into a single dict with four keys 'years', 'texts', 'features' and 'categories'
# each containing a list (with the same size), each list index refers to one entry of the bib files.
# Applies text vectorization techniques to get features .
class TextVectorizer:
    def __init__(self, vectorizer=TfidfVectorizer()):
        self.training_features = None
        self.testing_features = None
        self._vectorizer = vectorizer

    def extract_features(self, training_texts: list, testing_texts: list):
        self._vectorizer.fit(training_texts)
        self.training_features = self._vectorizer.transform(training_texts)
        self.testing_features = self._vectorizer.transform(testing_texts)


# Class that receives list of dict containing filtered data. Initialize text vectorization class with its configuration
# to generate new dataset with features
class DatasetGenerator:
    dataset = list()

    def __init__(self, filtered_training_set: list, filtered_testing_set: list):
        self.training_dataset = None
        self.testing_dataset = None
        self._filtered_training_set = filtered_training_set
        self._filtered_testing_set = filtered_testing_set

    # Receives a list with the text content of datasets, get features associated with each entry by applying text vectorization.
    @staticmethod
    def format_entry_set(dataset_lst) -> dict:
        # Formats to use boolean for selected entries (1) and non-selected entries (0)
        categories = [1 if text_data['category'] == 'selected' else 0 for text_data in dataset_lst]
        texts = [text_data['content'] for text_data in dataset_lst]
        years = [text_data['year'] for text_data in dataset_lst]
        titles = [text_data['title'] for text_data in dataset_lst]
        uuid = [text_data['uuid'] for text_data in dataset_lst]

        dataset = {
            'uuid': uuid,
            'titles': titles,
            'texts': texts,
            'categories': np.array(categories),
            'years': years
        }
        return dataset

    def execute(self):
        print("\t== [Executing DatasetGenerator] Vectorizing content and formating dataset ==")
        training_dataset = DatasetGenerator.format_entry_set(self._filtered_training_set)
        testing_dataset = DatasetGenerator.format_entry_set(self._filtered_testing_set)

        tf_idf_vectorizer = TextVectorizer(TfidfVectorizer(ngram_range=(1, 3), use_idf=True))
        tf_idf_vectorizer.extract_features(training_texts=training_dataset['texts'],
                                           testing_texts=testing_dataset['texts'])

        training_dataset['features'] = tf_idf_vectorizer.training_features
        testing_dataset['features'] = tf_idf_vectorizer.testing_features
        print('\nTotal number of features for TRAINING set =', training_dataset['features'].shape[1])
        print('Total number of features for TESTING set =', testing_dataset['features'].shape[1], end='\n')

        self.training_dataset = training_dataset
        self.testing_dataset = testing_dataset
        print("-----------------------------------------------------")


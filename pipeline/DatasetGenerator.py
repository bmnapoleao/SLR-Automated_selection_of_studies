# Author: Marcelo Costalonga
# (code adapted from: https://github.com/watinha/automatic-selection-slr/blob/master/pipeline/__init__.py)

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Class that converts lists of dicts into a single dict with four keys 'years', 'texts', 'features' and 'categories'
# each containing a list (with the same size), each list index refers to one entry of the bib files.
# Applies text vectorization techniques to get features .
class TextVectorizer:
    dataset_testing = list()
    dataset_training = list()

    def __init__(self, vectorizer=TfidfVectorizer()):
        self._vectorizer = vectorizer

    # Receives a list with the text content of datasets, get features associated with each entry by applying text
    # vectorization.
    def format_entry_set(self, dataset_lst):
        # Formats to use boolean for selected entries (1) and non-selected entries (0)
        categories = [1 if text_data['category'] == 'selecionado' else 0 for text_data in dataset_lst]
        texts = [text_data['content'] for text_data in dataset_lst]
        years = [text_data['year'] for text_data in dataset_lst]
        features = self._vectorizer.fit_transform(texts)
        dataset = {
            'texts': texts,
            'features': features,
            'categories': np.array(categories),
            'years': years
        }
        return dataset

# Class that receives list of dict containing filtered data. Initialize text vectorization class with its configuration
# to generate new dataset with features
class DatasetGenerator:
    testing_dataset = list()
    training_dataset = list()

    def __init__(self, filtered_testing_set: list, filtered_training_set: list):
        self._testing_set = filtered_testing_set
        self._training_set = filtered_training_set

    def execute(self):
        vectorizer = TextVectorizer(TfidfVectorizer(ngram_range=(1, 3), use_idf=True))
        self.testing_dataset = vectorizer.format_entry_set(self._testing_set)
        self.training_dataset = vectorizer.format_entry_set(self._training_set)
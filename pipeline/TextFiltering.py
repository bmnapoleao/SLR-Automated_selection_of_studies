# Author: Marcelo Costalonga
# (code adapted from: https://github.com/watinha/automatic-selection-slr/blob/master/pipeline/preprocessing.py)

from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# This module contains classes related to text filtering for applying NLP techniques (such as Lemmatization, StopWords,
# Tokenization) on the texts from the bib files, removing irrelevant words, typos and etc.
# Each NLP strategy uses a different configuration

# Composite Pattern for applying text filtering techniques
class TextFilterComposite:
    def __init__ (self, filters):
        self._filters = filters

    def _filter (self, tokens):
        result = tokens
        for f in self._filters:
            result = f.filter(result)
        return (' ').join(result)

    # Applies tokenization technique on texts' content (title and abstract)
    def apply_filters(self, text_list: list):
        result = []
        for text in text_list:
            # TODO#: Check different configurations
            # See if other tokens should be considered (noted that some examples contained chars like '$')
            tokens = word_tokenize(text['content'])
            filtered_text = self._filter(tokens)
            result.append({
                'title': text['title'],
                'content': filtered_text.lower(),
                'category': text['category'],
                'year': text['year'],
                'uuid': text['uuid']
            })
        return result

# NLP technique for configuring and using Lemmatization
class LemmatizerFilter:
    def __init__ (self):
        print('===== Configure the lemmatizer =====')
        self._lemmatizer = WordNetLemmatizer()

    def filter (self, tokens):
        # TODO#: Check different configurations
        tags = pos_tag(tokens)
        filtered_result = [
            self._lemmatizer.lemmatize(token[0], pos=token[1][0].lower())
            if token[1][0].lower() in ('a', 'n', 'v', 'r')
            else self._lemmatizer.lemmatize(token[0])
            for token in tags
        ]
        return filtered_result

# NLP technique for configuring and using StopWords
class StopWordsFilter:
    def __init__ (self):
        print('===== Configuring stop words removal =====')

    def filter (self, tokens):
        filtered_result = [word for word in tokens if not word.lower() in stopwords.words('english')]
        return filtered_result

# Class that receives valid data extracted from bib files and converts to filtered data by applying calling other
# classes to apply NLP techniques
class TextFilter:
    filtered_dataset = list()

    def __init__(self, training_set: list, testing_set: list):
        # self._dataset = training_set + testing_set # TODO#: Single dataset vs Two datasets
        self.filtered_training_set = None
        self.filtered_testing_set = None
        self._training_set = training_set
        self._testing_set = testing_set

    def execute(self):
        print("\t== [Executing TextFilter] Applying text filters ==")
        filters = [LemmatizerFilter(), StopWordsFilter()]
        text_filter_composite = TextFilterComposite(filters)
        # self.filtered_dataset = text_filter_composite.apply_filters(self._dataset) # TODO#: Single dataset vs Two datasets
        self.filtered_training_set = text_filter_composite.apply_filters(self._training_set)
        self.filtered_testing_set = text_filter_composite.apply_filters(self._testing_set)
        print("-----------------------------------------------------")


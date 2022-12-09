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
            tokens = word_tokenize(text['content'])
            filtered_text = self._filter(tokens)
            result.append({
                'content': filtered_text.lower(),
                'category': text['category'],
                'year': text['year']
            })
        return result

# NLP technique for configuring and using Lemmatization
class LemmatizerFilter:
    def __init__ (self):
        print('===== Configure the lemmatizer =====')
        self._lemmatizer = WordNetLemmatizer()

    def filter (self, tokens):
        tags = pos_tag(tokens)
        return [self._lemmatizer.lemmatize(token[0], pos=token[1][0].lower())
                    if token[1][0].lower() in ('a', 'n', 'v', 'r')
                    else self._lemmatizer.lemmatize(token[0])
                    for token in tags]

# NLP technique for configuring and using StopWords
class StopWordsFilter:
    def __init__ (self):
        print('===== Configuring stop words removal =====')

    def filter (self, tokens):
        return [word for word in tokens if not word.lower() in stopwords.words('english')]

# Class that receives valid data extracted from bib files and converts to filtered data by applying calling other
# classes to apply NLP techniques
class TextFilter:
    filtered_testing_set = list()
    filtered_training_set = list()

    def __init__(self, testing_set: list, training_set: list):
        self._testing_set = testing_set
        self._training_set = training_set

    def execute(self):
        filters = [LemmatizerFilter(), StopWordsFilter()]
        text_filter_composite = TextFilterComposite(filters)
        self.filtered_testing_set = text_filter_composite.apply_filters(self._testing_set)
        self.filtered_training_set = text_filter_composite.apply_filters(self._training_set)

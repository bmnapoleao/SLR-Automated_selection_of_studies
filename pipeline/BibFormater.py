# Author: Marcelo Costalonga

import codecs, bibtexparser
import random
import os
from copy import deepcopy

# Class to validate bib files of the Training and Testing sets, checking missing keys or duplicated entries
class BibFormater:
    training_set = list()
    testing_set = list()
    _are_entries_valid = True
    bibs_examples_dir = '/bibs-other-exemples'  # TODO: Change to not use hardcoded fullpath
    bibs_examples_formated_dir = os.path.join(bibs_examples_dir, 'bibs-formated-data')
    bibs_examples_original_dir = os.path.join(bibs_examples_dir, 'bibs-original-format')

    def __init__(self, random_seed: int, kfs: int, verbose: bool=False):
        print('Running program for: seed={} and kFs={}'.format(random_seed, kfs))
        print("\n-----------------------------------------------------")
        self.random_seed = random_seed
        self.verbose_looger = verbose
        print('===== Formating input files =====')

    def _check_required_info(self, entry):
        is_entry_valid = True
        if ('abstract' not in entry) or entry['abstract'] == None or len(entry['abstract']) == 0:
            print("\tMissing abstract:", entry)
            is_entry_valid = False
        if ('year' not in entry) or entry['year'] == None or len(entry['year']) == 0:
            print("\tMissing year:", entry)
            is_entry_valid = False
        if ('title' not in entry) or entry['title'] == None or len(entry['title']) == 0:
            print("\tMissing title:", entry)
            is_entry_valid = False
        return is_entry_valid

    def reading_single_bib(self):
        texts_list = list()
        titles_list = list()
        duplicated_titles = dict()

        print("HERE:")
        print(self.bibs_examples_dir)
        print(self.bibs_examples_formated_dir)
        print(self.bibs_examples_original_dir)

        for single_file in os.listdir(self.bibs_examples_original_dir):
            fullpath = os.path.join(self.bibs_examples_original_dir, single_file)
            print("FULL PATH:", fullpath) # TODO: remove

            print("Reading file:", fullpath)
            with codecs.open(fullpath, 'r', encoding='utf-8') as bib_file:
                bib_parsed = bibtexparser.load(bib_file)
                for bib_index, entry in enumerate(bib_parsed.entries, start=0):

                    is_entry_valid = self._check_required_info(entry)
                    category = 'selected' if entry['inserir'] == 'true' else 'removed'

                    if (is_entry_valid):
                        abstract = entry['abstract']
                        title = entry['title']
                        year = entry['year']
                        content = u'%s\n%s' % (title, abstract)

                        if (title not in titles_list):
                            titles_list.append(title)
                            content = content.split('\n')[0]
                            texts_list.append({
                                'title': title,
                                'content': content,
                                'category': category,
                                'year': int(year)
                            })
                        else:
                            if title in duplicated_titles:
                                duplicated_titles[title] += 1
                            else:
                                duplicated_titles[title] = 2
                    else:
                        self._are_entries_valid = False
                bib_file.close()

                # if (len(duplicated_titles) > 0):
                #     print("\n\tFound {} duplicated entries on set: {}".format(len(duplicated_titles), set_name))
                #     print(
                #         "\tEach of the following entries were found Nx times (number the same title was found) and were ignored:")
                #     for title in duplicated_titles:
                #         print("\t\t({}x) - {}".format(duplicated_titles[title], title))
                print("\tNumber of entries:", len(texts_list))
                return texts_list

    @staticmethod
    def get_dataset_years(dataset):
        years = dict()
        for i in dataset:
            if i['year'] not in years:
                years[i['year']] = [i]
            else:
                years[i['year']].append(i)
        print('\tTotal number of years: {}'.format(len(years)))
        for i in years:
            print('\t\t [{}] - {} bibs'.format(i, len(years[i])))
        return years

    def execute(self):
        tmp_training_set_included = list()
        tmp_training_set_excluded = list()
        tmp_testing_set_included = list()
        tmp_testing_set_excluded = list()

        # To shuffle list with fixed seed
        all_sets = self.reading_single_bib()
        random.Random(self.random_seed).shuffle(all_sets)

        print("ALL SETS ORDER:")
        for i in range(10):
            print('\t {} - {}'.format(i, all_sets[i]))

        print("\n\t Selected entries:")
        counter = 1
        for i in all_sets:
            if i['category'] == 'selected':
                if (self.verbose_looger):
                    print('\t\t', counter, '-', i['category'], i['title'][0:60])
                counter += 1
                if len(tmp_testing_set_included) < 21:
                    tmp_testing_set_included.append(i)
                else:
                    tmp_training_set_included.append(i)

        print("\n\t Excluded entries:")
        counter = 1
        for i in all_sets:
            if i['category'] == 'removed':
                if (self.verbose_looger):
                    print('\t\t', counter, '-', i['category'], i['title'][0:60])
                counter += 1
                if len(tmp_training_set_excluded) < 104:
                    tmp_training_set_excluded.append(i)
                else:
                    tmp_testing_set_excluded.append(i)

        self.training_set = tmp_training_set_included + tmp_training_set_excluded
        self.training_set = sorted(self.training_set, key=lambda d: d['year'])

        self.testing_set = tmp_testing_set_included + tmp_testing_set_excluded
        self.testing_set = sorted(self.testing_set, key=lambda d: d['year'])

        print("\nTraining dataset lenght:", len(self.training_set))
        print("Testing dataset lenght:", len(self.testing_set))

        if (self.verbose_looger):
            print("\n\t TEST set:")
            counter = 1
            for i in self.testing_set:
                cat = 'REM' if i['category'] == 'removed' else 'SEL'
                print('\t\t {}) {} -'.format(counter, cat), i['title'][0:60])
                counter += 1

            print("\n\t TRAING set:")
            counter = 1
            for i in self.training_set:
                cat = 'REM' if i['category'] == 'removed' else 'SEL'
                print('\t\t {}) {} -'.format(counter, cat), i['title'][0:60])
                counter += 1

        # INFO ABOUT YEARS
        print('\n\t TRAINING SET YEARS:')
        training_set_years = BibFormater.get_dataset_years(self.training_set)
        print('\n\t TESTING SET YEARS:')
        testing_set_years = BibFormater.get_dataset_years(self.testing_set)

        # # # Using only small sample of datasets
        # file_path_excluded_testing = os.path.join(os.getcwd(), 'bibs-sample/Testing set - Excluded.bib')
        # file_path_included_testing = os.path.join(os.getcwd(), 'bibs-sample/Testing set - Included.bib')
        # file_path_excluded_training = os.path.join(os.getcwd(), 'bibs-sample/Training set - Excluded.bib')
        # file_path_included_training = os.path.join(os.getcwd(), 'bibs-sample/Training set - Included.bib')
        #
        # testing_excluded = self.validate_bib_file(file_path_excluded_testing, 'Testing-Excluded', False)
        # testing_included = self.validate_bib_file(file_path_included_testing, 'Testing-Included', True)
        # training_excluded = self.validate_bib_file(file_path_excluded_training, 'Training-Excluded', False)
        # training_included = self.validate_bib_file(file_path_included_training, 'Training-Included', True)
        #
        # if self.validate_datasets(testing_excluded, testing_included, 'Testing Set'):
        #     self.testing_set = testing_excluded + testing_included
        #     self.testing_set = sorted(self.testing_set, key=lambda d: d['year'])
        # if self.validate_datasets(training_excluded, training_included, 'Training Set'):
        #     self.training_set = training_excluded + training_included
        #     self.training_set = sorted(self.training_set, key=lambda d: d['year'])
        # self.validate_datasets(self.testing_set, self.training_set, 'Training and Testing Sets')
        print("-----------------------------------------------------")

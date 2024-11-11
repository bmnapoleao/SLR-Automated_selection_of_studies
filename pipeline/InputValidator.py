# Author: Marcelo Costalonga

import os
import codecs
import bibtexparser
from copy import deepcopy
from TestConfigurationLoader import TestConfiguration


# Class to validate bib files of the Training and Testing sets, checking missing keys or duplicated entries
class InputValidator:
    training_set = list()
    testing_set = list()
    _are_entries_valid = True

    def __init__ (self):
        print("\n-----------------------------------------------------")
        print('===== Validating input files =====')

    # Method to validade a .bib file and report any errors found and removes duplicates (entries with the same title)
    # Return list of dicts with uniques entries associated with each set
    def validate_bib_file(self, file_path: str, set_name: str, was_selected: bool):
        texts_list = list()
        titles_list = list()
        duplicated_titles = dict()

        print("Validating file:", file_path)
        with codecs.open(file_path, 'r', encoding='utf-8') as bib_file:
            db = bibtexparser.load(bib_file)
            for bib_index, entry in enumerate(db.entries, start=0):
                is_entry_valid = True
                category = 'selected' if was_selected == True else 'removed'
                if ('abstract' not in entry) or entry['abstract'] == None or len(entry['abstract']) == 0:
                    print("\tMissing abstract:", entry)
                    is_entry_valid = False
                if ('year' not in entry) or entry['year'] == None or len(entry['year']) == 0:
                    print("\tMissing year:", entry)
                    is_entry_valid = False
                if ('title' not in entry) or entry['title'] == None or len(entry['title']) == 0:
                    print("\tMissing title:", entry)
                    is_entry_valid = False

                if (is_entry_valid):
                    abstract = entry['abstract']
                    title = entry['title']
                    year = entry['year']
                    uuid = entry.get('uuid', None)
                    content = u'%s\n%s' % (title, abstract)

                    if (title not in titles_list):
                        titles_list.append(title)

                        # This should be used only if we want to consider studies that doesn't have an abstract field
                        # so we would evaluate the study only based on its title
                        # # content = content.split('\n')[0]

                        texts_list.append({
                            'title': title,
                            'content': content,
                            'category': category,
                            'year': int(year),
                            'uuid': uuid
                        })
                    else:
                        if title in duplicated_titles:
                            duplicated_titles[title] += 1
                        else:
                            duplicated_titles[title] = 2
                else:
                    self._are_entries_valid = False
            bib_file.close()

            if (len(duplicated_titles) > 0):
                print("\n\tFound {} duplicated entries on set: {}".format(len(duplicated_titles), set_name))
                print("\tEach of the following entries were found Nx times (number the same title was found) and were ignored:")
                for title in duplicated_titles:
                    print("\t\t({}x) - {}".format(duplicated_titles[title], title))
            print("\tNumber of entries:", len(texts_list))
            return texts_list

    # Create list with duplicated entries by comparing difference between list and set with unique entries
    @staticmethod
    def get_duplicates(lst1: list, set2: set):
        lst_copy = deepcopy(lst1)
        for i in set2:
            if i in lst_copy:
                lst_copy.remove(i)
        return lst_copy

    # Check if there are no duplicated entry among included/excluded dataset
    def validate_datasets(self, dataset1: list, dataset2: list, dataset_type: str):
        titles1 = [t['title'] for t in dataset1]
        titles2 = [t['title'] for t in dataset2]
        try:
            unique_titles = set(titles1 + titles2)
            merged_titles = list(titles1 + titles2)
            assert len(merged_titles) == len(unique_titles)
        except AssertionError:
            print("\n{} sets have {} duplicated entries!".format(dataset_type, len(merged_titles) - len(unique_titles)))
            duplicated_entries = InputValidator.get_duplicates(merged_titles, unique_titles)
            print("The following entries appear both as included and excluded in the {} set:\n".format(dataset_type),
                  duplicated_entries)
            self._are_entries_valid = False
            return False
        return True

    def is_valid(self):
        return self._are_entries_valid

    # Validates the four files of ../bibs/ and initialize data structures
    def execute(self):
        # Loading dataset configuration
        dataset_option = TestConfiguration().get_dataset_type()

        # TODO: EXPERIMENT - Change datasets
        if dataset_option == 0:
            # # Using small fake dataset just to test pipeline execution
            print('\n\tUSING SMALL DATASET')
            file_path_excluded_testing = os.path.join(os.getcwd(), 'bibs-small-set/Testing set - Excluded.bib')
            file_path_included_testing = os.path.join(os.getcwd(), 'bibs-small-set/Testing set - Included.bib')
            file_path_excluded_training = os.path.join(os.getcwd(), 'bibs-small-set/Training set - Excluded.bib')
            file_path_included_training = os.path.join(os.getcwd(), 'bibs-small-set/Training set - Included.bib')

        elif dataset_option == 1:
            # # Using original large dataset (testing set way bigger than training)
            print('\n\tUSING ORIGINAL DATASET')
            file_path_excluded_testing = os.path.join(os.getcwd(), 'bibs/Testing set - Excluded.bib')
            file_path_included_testing = os.path.join(os.getcwd(), 'bibs/Testing set - Included.bib')
            file_path_excluded_training = os.path.join(os.getcwd(), 'bibs/Training set - Excluded.bib')
            file_path_included_training = os.path.join(os.getcwd(), 'bibs/Training set - Included.bib')

        elif dataset_option == 2:
            # # Using inverted large dataset (training set way bigger than testing)
            print('\n\tUSING INTERVETED DATASET')
            file_path_excluded_training = os.path.join(os.getcwd(), 'bibs/Testing set - Excluded.bib')
            file_path_included_training = os.path.join(os.getcwd(), 'bibs/Testing set - Included.bib')
            file_path_excluded_testing = os.path.join(os.getcwd(), 'bibs/Training set - Excluded.bib')
            file_path_included_testing = os.path.join(os.getcwd(), 'bibs/Training set - Included.bib')
        else:
            print("\n[ERROR-EnvFile] Invalid dataset option")
            raise Exception

        testing_excluded = self.validate_bib_file(file_path_excluded_testing, 'Testing-Excluded', was_selected=False)
        testing_included = self.validate_bib_file(file_path_included_testing, 'Testing-Included', was_selected=True)
        training_excluded = self.validate_bib_file(file_path_excluded_training, 'Training-Excluded', was_selected=False)
        training_included = self.validate_bib_file(file_path_included_training, 'Training-Included', was_selected=True)

        if self.validate_datasets(testing_excluded, testing_included, 'Testing Set'):
            self.testing_set = testing_excluded + testing_included
            self.testing_set = sorted(self.testing_set, key=lambda d: d['year'])
        if self.validate_datasets(training_excluded, training_included, 'Training Set'):
            self.training_set = training_excluded + training_included
            self.training_set = sorted(self.training_set, key=lambda d: d['year'])
        self.validate_datasets(self.testing_set, self.training_set, 'Training and Testing Sets')

        try:
            assert self.is_valid() == True
        except AssertionError:
            print("\nPlease correct the errors found on the bib files before executing the program again.")
            exit(0)
        print("-----------------------------------------------------")


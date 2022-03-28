"""
Script based on the stylometry package by Jeff Potter and adjusted for its use on German-written texts.

Requires a predownloaded set of books divided into female/male authors (generated by get-books.sh).
When that is available, the script only requires the specification of which group of authors (male or female) the
features are needed of.
"""

from __future__ import division
from nltk.probability import FreqDist
import numpy as np
import os
import sys
import argparse
import spacy
from spacy.tokenizer import Tokenizer
import json
import pandas as pd

##########
# Initializing model
##########
print('Initializing SpaCy model')
nlp = spacy.load("de_core_news_sm")
tokenizer = Tokenizer(nlp.vocab)


# Defining helper functions
def sent_splitting(nlp_doc: spacy.tokens.doc.Doc):
    """
    Function to generate a list containing all the sentences of a document as split by the SpaCy parser.

    :param nlp_doc: SpaCy document
    :return: list['sentence1', 'sentence2', ...]
    """

    sentences = [sent.text for sent in nlp_doc.sents]
    return sentences

##########
# Define stylometrics extractor
##########


print('Defining stylometrics extractor')


class StyloDocument(object):
    def __init__(self, path, bookid, author='Unknown'):
        """
        We initialize our class for our document. The variable we declare here
        will be the attributes of the object we create (= the analyzed document).
        These correspond to basic features of our text that will help us measure
        different, more specific features later on.
        """
        # Reading the document and defining the author and file name
        self.doc_path = os.path.join(path, str(bookid)+'.txt')
        self.doc = open(self.doc_path, "r", encoding='utf-8', errors='replace').read()
        self.author = author
        self.file_name = bookid

        # We create a SpaCy Document based on our document and extract the tokens
        self.nlp_doc = nlp(self.doc)
        self.tokens = tokenizer(self.doc)

        # We create a string with our text with the lemmas of each token instead
        # of the token itself to account for German declension and other factors
        # that may influence how the same "word" has different forms.

        self.text = [token.lemma_ for token in self.nlp_doc]
        self.fdist = FreqDist(self.text)

        # Extracting sentences, characters per sentence, words per sentence, n of
        # paragraphs and the number of words per paragraph

        self.sentences = sent_splitting(self.nlp_doc)
        self.sentence_chars = [len(sent) for sent in self.sentences]
        self.sentence_word_length = [len(sent.split()) for sent in self.sentences]
        self.paragraphs = [p for p in self.doc.split("\n\n") if len(p) > 0 and not p.isspace()]
        self.paragraph_word_length = [len(p.split()) for p in self.paragraphs]

    # Defining further methods for specific features

    def term_per_thousand(self, term):
        """
        Gets the frequency of a term or character per 1,000 words.

        Calculated as:

        term       X
        -----  = ------
          N       1000

        :param term:lexeme/character to be counted
        :return: float:frequency per 1,000 words
        """
        return (self.fdist[term] * 1000) / self.fdist.N()

    def mean_sentence_len(self):
        """
        Gets the mean sentence length of a text based on the number of words

        :return: float:mean sentence length
        """
        return np.mean(self.sentence_word_length)

    def std_sentence_len(self):
        """
        Gets the standard deviation of the length of all sentences in a text as measured
        by the number of words in each of them.

        :return: float:standard deviation
        """
        return np.std(self.sentence_word_length)

    def mean_paragraph_len(self):
        """
        Gets the mean paragraph length of a text based on the number of words in each of them.

        :return: float:mean paragraph length
        """
        return np.mean(self.paragraph_word_length)

    def std_paragraph_len(self):
        """
        Gets the standard deviation of the length of all paragraphs in a text,
        based on the number of words in each of them

        :return: float:standard deviation of the paragraph lengths
        """
        return np.std(self.paragraph_word_length)

    def mean_word_len(self):
        """
        Collects all tokens, gets their individual lengths (in characters) and returns the mean length for all of them.

        :return: float:mean word length (in characters)
        """

        words = set(self.tokens)
        word_chars = [len(word) for word in words]
        return sum(word_chars) / float(len(word_chars))

    def type_token_ratio(self):
        """
        Gets type:token ratio for the entire text.

        Calculated as:
            Total number of unique words (types)
                            ---
        Total number of words (tokens) in the text


        :return: float: type:token ratio
        """
        return (len(set(self.text)) / len(self.text)) * 100

    def unique_words_per_thousand(self):
        """
        Gets the frequency of types per 1,000 words.

                TTR
                ---
            100 * 1,000
                ---
        total of words in text

        :return: float: frequency of types per 1,000 words
        """
        return self.type_token_ratio() / 100.0 * 1000.0 / len(self.text)

    def document_len(self):
        """
        Gets the length of the document as measured by the total number of characters in it.

        :return: int: total number of characters
        """
        return sum(self.sentence_chars)

    def get_feats(self):
        """
        Feature getter for all the features extracted in the script, exported in a single list.

        :return: list: all features
        """
        feats = [
            self.author,
            self.file_name,
            self.type_token_ratio(),
            self.mean_word_len(),
            self.mean_sentence_len(),
            self.std_sentence_len(),
            self.mean_paragraph_len(),
            self.document_len(),
            self.term_per_thousand(','),
            self.term_per_thousand(';'),
            self.term_per_thousand('"'),
            self.term_per_thousand('!'),
            self.term_per_thousand(':'),
            self.term_per_thousand('-'),
            self.term_per_thousand('--'),
            self.term_per_thousand('und'),
            self.term_per_thousand('aber'),
            self.term_per_thousand('obwohl'),
            self.term_per_thousand('wenn'),
            self.term_per_thousand('dass'),
            self.term_per_thousand('mehr'),
            self.term_per_thousand('sagen'),
            self.term_per_thousand('rufen'),
            self.term_per_thousand('flüstern'),
            self.term_per_thousand('vielleicht'),
            self.term_per_thousand('dies'),
            self.term_per_thousand('sehr')
        ]
        return feats

    def text_output(self):
        """
        Prints an overview of all extracted features for a single text and returns it in a single string.

        :return: str: print-friendly overview of the features
        """
        output = f"""##############################################
      {'Name' : <20} : {self.file_name}
      >>> Phraseology Analysis <<<

      {'Lexical diversity' : <20} : {self.type_token_ratio()}
      {'Mean Word Length' : <20} : {self.mean_word_len()}
      {'Mean Sentence Length' : <20}  : {self.mean_sentence_len()}
      {'St.Dev Sentence Length' : <20}  : {self.std_sentence_len()}
      {'Mean Paragraph Length' : <20}  : {self.mean_paragraph_len()}
      {'Document Length' : <20}  : {self.document_len()}

      >>> Punctuation Analysis (per 1000 tokens) <<<

      {'Commas' : <20}  : {self.term_per_thousand(',')}
      {'Semicolons' : <20}  : {self.term_per_thousand(';')}
      {'Quotations' : <20}  : {self.term_per_thousand('"')}
      {'Exclamations' : <20}  : {self.term_per_thousand('!')}
      {'Colons' : <20}  : {self.term_per_thousand(':')}
      {'Hyphens' : <20}  : {self.term_per_thousand('-')} # m-dash or n-dash?
      {'Double Hyphens' : <20}  : {self.term_per_thousand('--')} # m-dash or n-dash?

      >>> Lexical Usage Analysis (per 1000 tokens) <<<

      {'und' : <20}  : {self.term_per_thousand('and')}
      {'aber' : <20}  : {self.term_per_thousand('aber')}
      {'obwohl' : <20}  : {self.term_per_thousand('obwohl')}
      {'wenn' : <20}  : {self.term_per_thousand('wenn')}
      {'dass' : <20}  : {self.term_per_thousand('dass')}
      {'mehr' : <20}  : {self.term_per_thousand('mehr')}
      {'sagen': <20}  : {self.term_per_thousand('sagen')},
      {'rufen' : <20} : {self.term_per_thousand('rufen')},
      {'flüstern': <20} : {self.term_per_thousand('flüstern')},
      {'vielleicht' : <20}  : {self.term_per_thousand('vielleicht')}
      {'diese/r/n/s' : <20}  : {self.term_per_thousand('dies')}
      {'sehr' : <20}  : {self.term_per_thousand('sehr')}
      """
        print(output)
        return output

    def export(self):
        """
        Generates a dictionary containing all the extracted features of a single text for easy feature retrieval.

        :return: dict: feature name : feature value pairs
        """
        exported_doc = {
            'author': self.author,
            'filename': self.file_name,
            'TTR': self.type_token_ratio(),
            'mean_word_len': self.mean_word_len(),
            'mean_sent_len': self.mean_sentence_len(),
            'std_sent_len': self.std_sentence_len(),
            'mean_p_len': self.mean_paragraph_len(),
            'doc_len': self.document_len(),

            'commas': self.term_per_thousand(','),
            'semicolons': self.term_per_thousand(';'),
            'quotations': self.term_per_thousand('"'),
            'exclamations': self.term_per_thousand('!'),
            'colons': self.term_per_thousand(':'),
            'hyphens': self.term_per_thousand('-'),
            'double_hyphens': self.term_per_thousand('--'),

            'und': self.term_per_thousand('und'),
            'aber': self.term_per_thousand('aber'),
            'obwohl': self.term_per_thousand('obwohl'),
            'wenn': self.term_per_thousand('wenn'),
            'dass': self.term_per_thousand('dass'),
            'mehr': self.term_per_thousand('mehr'),
            'sagen': self.term_per_thousand('sagen'),
            'rufen': self.term_per_thousand('rufen'),
            'flüstern': self.term_per_thousand('flüstern'),
            'vielleicht': self.term_per_thousand('vielleicht'),
            'dies': self.term_per_thousand('dies'),
            'sehr': self.term_per_thousand('sehr'),
        }
        return exported_doc


def extract_metrics(path, book_id, author, export_dir='.', export=False):
    """
    Instantiates a StyloDocument object given a specified directory containing files named according to their bookid
    and extracts all its features.
    Writes an individual .txt report with the extracted features and stores this file in the specified export directory
    if desired and returns the dictionary containing the features for the specific text.

    :param path: str:location of the texts' .txt files
    :param book_id: str, int or float: work's book ID as specified in Project Gutenberg
    :param author: str:Work's author's name
    :param export_dir: str: directory to store the individual feature reports, defaults to current directory
    :param export: bool:write individual feature report files yes/no
    :return: dictionary containing the text's features
    """
    print(f'Working on text {book_id}')
    style = StyloDocument(path=path, bookid=book_id, author=author)

    if export:
        with open(os.path.join(export_dir, str(book_id) + '.txt'), 'w') as individual:
            individual.write(style.text_output())

    return style.export()


def main(argv):
    # We initialize the parser for the arguments defined in our terminal
    parser = argparse.ArgumentParser(
        description="Stylometry feature extractor of German-written texts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        fromfile_prefix_chars='@')

    # Specifying which arguments we are going to require our user to enter through the terminal,
    # in this case we want our user to specify for which gender they want to collect the features from
    # and whether they wish to get individual .txt files with the features of each text or not.

    parser.add_argument('--gender_author', '-g',
                        help='specify the gender of the authors being analyzed',
                        required=True)
    parser.add_argument('--export', '-e',
                        help='get individual feature reports for each text',
                        action='store_true')

    # Collect the terminal arguments
    args = parser.parse_args(argv)

    # Control that the gender entered is binary since that's how our system works.
    # Exit the entire script if that's not the case as the script won't work otherwise.
    if args.gender_author not in ['female', 'male']:
        print('Invalid path allocation, please control gender flag and that you are using the default data location')
        sys.exit()

    # Specify the location of our texts
    data = os.path.join('..', 'collect_data', 'data', args.gender_author)
    print(f'Parsing books tagged as {args.gender_author} authors')

    # Specify and create the directory where we want to store our results
    dir_results = os.path.join('.', 'results')
    if not os.path.exists(dir_results):
        os.makedirs(dir_results)

    # Go through the metadata of the downloaded books to start extracting their features:
    print('Reading metadata')

    metadata_path = os.path.join(data, 'metadata.tsv')
    books = pd.read_csv(metadata_path, sep='\t', header=None)

    print('Extracting features...')

    # Alternative way of formulating lines 373-382:
    # analysis = dict(books.apply(lambda b: (b[0], extract_metrics(data, b[0], b[1])), axis=1))

    # We initialize a dictionary where we will store all our texts' ids and their corresponding features
    # and loop through the rows in our metadata file. In each loop, we extract the features for each text
    # and store them in the earlier defined dictionary.

    analysis = {}

    for index, entry in books.iterrows():
        bookid = entry[0]
        author = entry[1]
        analysis[bookid] = extract_metrics(path=data,
                                           book_id=bookid,
                                           author=author,
                                           export_dir=dir_results,
                                           export=args.export)

    # Once the loop is done, we store the now filled dictionary in a file that can be read as a dataframe
    with open(os.path.join(dir_results, f'stylometry_{args.gender_author}_authors.json'), 'w') as fp:
        json.dump(analysis, fp)

    print('DONE!')
    print(analysis)


if __name__ == "__main__":
    main(sys.argv[1:])

# Name: Saeed Baghershahi
# Student Number: 102501002
# Class: Deep Learning
# Assignment: Project: 01, Chapter: 09, Book: "Python Machine Learning By Example"

from nltk.corpus import names
print(names.words()[:10])
print(len(names.words()))

from nltk.tokenize import word_tokenize
sent = '''I am reading a book.
          It is Python Machine Learning By Example, 
          3rd edition.'''
print(word_tokenize(sent))

sent2 = 'I have been to U.K. and U.S.A.'
print(word_tokenize(sent2))

import spacy
nlp = spacy.load('en_core_web_sm')
tokens2 = nlp(sent2)
print([token.text for token in tokens2])

from nltk.tokenize import sent_tokenize
print(sent_tokenize(sent))

import nltk
tokens = word_tokenize(sent)
print(nltk.pos_tag(tokens))

nltk.help.upenn_tagset('PRP')
nltk.help.upenn_tagset('VBP')

print([(token.text, token.pos_) for token in tokens2])
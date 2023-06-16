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


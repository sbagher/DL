# Name: Saeed Baghershahi
# Student Number: 102501002
# Class: Deep Learning
# Assignment: Final Project, Phase Two: Data Cleaning & Test, Poet: Parvin E'tesami, Book: "Python Machine Learning By Example"

import random

with open('poems-utf8-same-weight-2.txt', 'w', encoding="utf-8") as f:
    lines = open('poems-utf8-same-weight.txt', 'r', encoding="utf-8").readlines()
    f.writelines(lines)
    random.shuffle(lines)
    f.writelines(lines)
    f.close

with open('poems-utf8-same-weight-4.txt', 'w', encoding="utf-8") as f:
    lines = open('poems-utf8-same-weight-2.txt', 'r', encoding="utf-8").readlines()
    f.writelines(lines)
    random.shuffle(lines)
    f.writelines(lines)
    f.close

with open('poems-utf8-same-weight-6.txt', 'w', encoding="utf-8") as f:
    lines = open('poems-utf8-same-weight-4.txt', 'r', encoding="utf-8").readlines()
    f.writelines(lines)
    lines = open('poems-utf8-same-weight-2.txt', 'r', encoding="utf-8").readlines()
    random.shuffle(lines)
    f.writelines(lines)
    f.close

with open('poems-utf8-same-weight-8.txt', 'w', encoding="utf-8") as f:
    lines = open('poems-utf8-same-weight-4.txt', 'r', encoding="utf-8").readlines()
    f.writelines(lines)
    random.shuffle(lines)
    f.writelines(lines)
    f.close

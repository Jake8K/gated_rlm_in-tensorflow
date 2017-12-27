"""
Tokenize a file and remove trailing spaces
by: Jacob Karcz
11.18.2017
"""
#uncomment tensorflow import if using gfile

import numpy
import cPickle as pkl

import sys
import os
import fileinput
import time
import re
from datetime import datetime
#from tensorflow.python.platform import gfile


from collections import OrderedDict

def corpus_tokenizer(corpus, destination):
    words = []
    #with gfile.GFile(destination, mode="wb") as tokenized_file:
    with open(destination, mode="wb") as tokenized_file:
        with open(corpus, mode='rb') as fp:
            for line in fp:
                for space_separated_fragment in line.strip().split():
                    #shorter than the if bellow https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
                    #if '.' or ',' or ';' or ':' or '?' or '!' or '\'' or '\'s' or '\'ve' '\'ll' in space_separated_fragment:
                    #tokenized_file.write(words.extend(re.split(" ", space_separated_fragment)) + b"\n")
                    #words = clean_str(space_separated_fragment)+b' '
                    #l = l + words
                    tokenized_file.write(clean_str(space_separated_fragment)+b' ')
                tokenized_file.write('\n')

def clean_str(string):
    """
        Tokenization/string cleaning for all datasets except for SST.
        Original mostly taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data
        more info: https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
        """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string) #what does this one do?
    return string.strip()

def erase_trailing(filename):
    destination = filename.replace('.txt.tmp', '_tokenized.txt')
    with open(destination, mode="wb") as clean_file:
    #with gfile.GFile(destination, mode="wb") as clean_file:
        with open(filename, mode='rb') as fp:
            for line in fp:
                clean_file.write(line.strip()+'\n')


def main(filenames):
    for filename in filenames:
        tic = time.time()
        print '==> Processing', filename
        corpus_tokenizer(filename, filename+".tmp" )
        erase_trailing(filename+".tmp")
        toc = time.time()
        t = toc - tic
        print '\tFinished Processing {} in {} seconds'.format(filename, t)
        os.remove(os.path.join(filename+".tmp"))

if __name__ == '__main__':
	filenames = sys.argv[1:]
	assert len(filenames) > 0, "please specify at least one filename."
	main(filenames)

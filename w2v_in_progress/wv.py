"""Based on the code from: https://github.com/ryankiros/skip-thoughts/blob/master/skipthoughts.py"""
from __future__ import print_function
import os
import math
import random
import string
import zipfile
import collections
import random
import zipfile
import datetime

import tensorflow as tf
import numpy as np

from matplotlib import pylab
from six.moves import range
from six.moves.urllib.request import urlretrieve

from collections import OrderedDict, defaultdict, Counter, deque
from random import shuffle

import sys

def loadMovieQAFiles():
    """
    Loads words from each file into a dictionary.
    Returns:
        tuple(map[filename] -> listOfWordsInFile, allWords)
    """
    path = '../MovieQA/story/plot'
    allWords = []
    movieToWordsDict = {}
    listOfFiles = os.listdir(path)
    for filename in listOfFiles:
        with open(path + '/' + filename) as file:
            paragraph = file.read()
            listOfWords = splitToWords(paragraph)
            movieToWordsDict[filename] = listOfWords
            allWords.extend(listOfWords)
    return (movieToWordsDict, allWords)  

def splitToWords(paragraph):
    """
    Given a paragraph of text, remove all punctuation/capitalization
    and place all words into a list.
    """
    listOfWords = []
    paragraph = paragraph.lower()
    paragraph = paragraph.replace('\n', ' ')
    for char in ",:;()!?.":
        paragraph = paragraph.replace(char, '')
    listOfWords = paragraph.split()
    return listOfWords

def updateVocabularyCounter(movieToWordsDict, vocabularyCounter):
    '''Add words to vocabularyCounter and update word frequency'''
    for movie in movieToWordsDict:
        for word in movieToWordsDict[movie]:
            if word in vocabularyCounter:
                vocabularyCounter[word] +=1
            else:
                vocabularyCounter[word] = 1
    return vocabularyCounter

def mapWordsToIndices(frequentWords, vocabularyCounter):
    # Index in 1-hot vector
    oneHotIndicesMap = {}
    uniqueIndex = 1
    for count, word in enumerate(vocabularyCounter):
        if word in frequentWords:
            oneHotIndicesMap[word] = uniqueIndex
            uniqueIndex += 1
    return oneHotIndicesMap

#NOTE TO SELF: FIX GENERATE_BATCH -----------------
def generateBatch(num_skips, batchSize, data, start_index):
    '''Generates a training batch from the given file'''
    #Note assumes that num_skips will also be the max distance from the current word
    #Data is the one-hot index representation of the words in the vocabularyCounter
    #num_skips is number of skips to do from each center word
    #start index is the position in clippedAndShuffledIndices_array
    assert batchSize%num_skips == 0 #Ensure batch size is divisible by number of skips
    assert num_skips%2 ==0
    windowSize = num_skips + 1
    num_center_words = batchSize/num_skips
    center_word_indices = clippedAndShuffledIndices[start_index:start_index+num_center_words]
    # Center words for current training batch
    batch = np.ndarray(shape=(batchSize), dtype=np.int32)
    # Labels for current training batch
    labels = np.ndarray(shape=(batchSize), dtype=np.int32)
    for currCenterIndex in range(num_center_words):
        # Fill training batch/labels
        center_word = center_word_indices[currCenterIndex]
        for halfSkipDistance in range(num_skips/2):
            currCenterWordPredictLeftIndex = num_skips*currCenterIndex + 2*halfSkipDistance
            currCenterWordPredictRightIndex = currCenterWordPredictLeftIndex + 1
            batch[currCenterWordPredictLeftIndex] = data[center_word]
            batch[currCenterWordPredictRightIndex] = data[center_word]
            # Predict Words halfSkipDistance to the left and right of center_word
            skipAmount = halfSkipDistance + 1
            labels[currCenterWordPredictLeftIndex] = data[center_word - skipAmount]
            labels[currCenterWordPredictRightIndex] = data[center_word + skipAmount]
    return (batch,labels)


if __name__ == "__main__":
    startTime = datetime.datetime.now()
    # TODO: Change into classes
    vocabularyCounter = Counter() # Map of word to # instances 
    (movieToWordsDict, allWords) = loadMovieQAFiles()
    vocabularyCounter = updateVocabularyCounter(movieToWordsDict, vocabularyCounter)
    VOCABULARY_SIZE = 10000
    frequentWordsCounter = vocabularyCounter.most_common(VOCABULARY_SIZE)
    frequentWords = []
    #Take VOCABULARY_SIZE most common words
    for word, count in frequentWordsCounter:
        frequentWords.append(word)
    # Map all frequent words to unique indices, and non-frequent words to 0
    oneHotIndicesMap = mapWordsToIndices(frequentWords, vocabularyCounter)

    # Data contains indices for all frequent words
    # and will be split into the respective train, validation, test sets
    data = []
    for word in allWords:
        if word in oneHotIndicesMap:
            data.append(oneHotIndicesMap[word])
        else:
            data.append(0)

    #Graph parameters
    windowSize = 4 # number of surround words to predict
    num_skips = 8
    batchSize = 128 # used in batch gradient descent
    embedding_dim = 128 # Embedding vector dimension
    start_index = 0
    num_sampled = 64 # Negative examples

    # Select possible center words and shuffle
    # Holds the center word indices for examples
    # Clip off ends to avoid going out of range for window

    # FIXME: Emily, here you forgot to account for skipping, so it goes out of range.
    clippedAndShuffledIndices = list(range(windowSize, len(allWords)-windowSize)) 

    shuffle(clippedAndShuffledIndices) # Holds indices of center words

    # Split (70% training, 20% test, 10% validation)
    datasetSize = len(clippedAndShuffledIndices)

    trainSetSize = int(math.floor(datasetSize*0.7))
    testSetSize = int(math.floor(datasetSize*0.2))
    validSetSize = datasetSize - (trainSetSize + testSetSize)

    trainSet = data[0: trainSetSize]
    testSet = data[trainSetSize: (trainSetSize + testSetSize)]
    validSet = data[(trainSetSize+testSetSize):] 

    # Initialize Tensorflow Graph
    graph = tf.Graph()
    with graph.as_default(),tf.device('/cpu:0'):
        # Initial loss
        loss = 0
        # The training data (center words) and training labels (predictions)
        train_data = tf.placeholder(tf.int32, shape=[batchSize])
        train_label = tf.placeholder(tf.int32, shape=[batchSize,1]) #Check shape (num_skips vs 1)
        embeddings = tf.Variable(tf.random_uniform([VOCABULARY_SIZE, embedding_dim], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_data)
        weights = tf.Variable(tf.truncated_normal([VOCABULARY_SIZE, embedding_dim], stddev=1.0 / math.sqrt(embedding_dim)))
        # DOUBLE CHECK BIASES
        biases = tf.Variable(tf.zeros([VOCABULARY_SIZE]))
        # Save model variables
        loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights, biases, embed, train_label, num_sampled, VOCABULARY_SIZE))
        # save = tf.train.Saver()
        # Gradient Descent
        optimizer = tf.train.AdagradOptimizer(0.01).minimize(loss)
        # Cosine Similarity
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings),1, keep_dims=True))
        normalized_embeddings = embeddings/norm

        # For plotting the loss
        loss_values = []

        # Begin Training
        with tf.Session(graph=graph) as session:
            # Initialize variables
            tf.initialize_all_variables().run()
            print('Variables initialized')
            num_steps = 100001
            start_index = 0 # iterate through the clippedAndShuffledIndices
            total_loss = 0
            for step in range(num_steps):
                batch_data, batch_labels = generateBatch(num_skips, batchSize, data, start_index)
                batch_labels = np.reshape(batch_labels,(128,1))
                feed_dict = {train_data: batch_data, train_label: batch_labels}
                _,l = session.run([optimizer,loss], feed_dict = feed_dict)
                total_loss += l
                # Estimate loss
                if ((step % 1000) == 0) and (step != 0):
                    loss_values.append(total_loss/1000)
                    total_loss = 0
                # Save data?
                # FIXME: EMILY, this is a bug, it's always training the same words
                # since the start index are the same always
                # I tried fixing your code below but it no longer works if I do += 1
                start_index =+ 1
            embeddings = normalized_embeddings.eval()
            print("Completed")
            pylab.ylabel("Loss")
            pylab.xlabel("Step #")
            pylab.plot(np.arange(1,100000, 1001),loss_values)
            pylab.show()            
    endTime = datetime.datetime.now()
    elapsedTime = endTime - startTime
    hours, remainder = divmod(elapsedTime.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    totalDays = elapsedTime.days
    print('Days: ' + str(totalDays) +  " hours: " + str(hours) + ' minutes: ' + str(minutes) +  ' seconds: ' + str(seconds))


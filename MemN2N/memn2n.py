import os
import math
import random
import string
import zipfile
import collections
import random

import tensorflow as tf
import numpy as np
import sys

import re # Split by tabs
import string # To remove punctuation

from matplotlib import pylab
#from sklearn.manifold import TSNE

from collections import OrderedDict, defaultdict, Counter, deque
from random import shuffle

class BabiParser(object):
    def __init__(self):
        self.fileNames = ["qa1_single-supporting-fact_train.txt"]
        #self.fileNames = ["temp.txt"]
        self.vocabularyToIndex = {}
        self.numWords = 0
        self.numStory = 0
        self.maxSentencePerStory = 0
        self.numQuestion = 0
        # Parse the vocabulary
        self.parseBabiTaskVocabulary()
        # Parse the vectors to get X, q, a (input, question, answer) sentence vectors
        self.X, self.q, self.a = self.parseBabiTaskVectors()
        print 'numWords:', self.numWords
        print 'numStory:', self.numStory
        print 'maxSentencePerStory:', self.maxSentencePerStory
        print 'numQuestion:', self.numQuestion
        print 'Input Shape:', self.X.shape
        print 'Question Shape:', self.q.shape
        print 'Answer Shape:', self.a.shape
        '''
        print self.X
        print self.q
        print self.a 
        '''
    def insertVocabulary(self, word):
        if word in self.vocabularyToIndex:
            return
        self.vocabularyToIndex[word] = self.numWords
        self.numWords += 1
        return

    def getSentenceVector(self, sentence):
        rowVector = list()
        for word in sentence.split():
            rowVector.append(self.vocabularyToIndex[word])
        sentenceVec = self.convertToOneHot(np.array(rowVector))
        return sentenceVec

    def convertToOneHot(self, rowVector):
        oneHotVectors = np.eye(self.numWords)[rowVector]
        return np.sum(oneHotVectors, 0)

    def parseBabiTaskVocabulary(self):
        """
        First loop is to assign each word into vocabulary
        Second loop is to create the matrix
        """
        uniqueIndex = 0
        vocabulary = []
        for currFile in self.fileNames:
            numSentencePerStory = 0 
            fd = open("en/" + currFile, "r")
            for currLine in fd.readlines():
                Id = currLine.split(' ')[0]
                # Start of a new story
                if Id == "1":
                    self.numStory += 1
                    numSentencePerStory = 0
                # A question
                if '\t' in currLine:
                    self.numQuestion += 1
                    # Need to close the current X
                    lineSplit = re.split(r'\t+', currLine)
                    # Don't include the index
                    question = lineSplit[0].split(' ', 1)[1].strip()
                    question = question.translate(None, string.punctuation)
                    answer = lineSplit[1].strip()
                    for currWord in question.split(' '):
                        self.insertVocabulary(currWord)
                    for currWord in answer.split(' '):
                        self.insertVocabulary(currWord)
                # A Sentence
                else:
                    numSentencePerStory += 1
                    self.maxSentencePerStory = max(self.maxSentencePerStory, numSentencePerStory)
                    sentence = currLine.split(' ', 1)[1].strip()
                    sentence = sentence.translate(None, string.punctuation)
                    for currWord in sentence.split(' '):
                        self.insertVocabulary(currWord)
        # Done inserting all word
        print self.numWords

    def parseBabiTaskVectors(self):
        """
        Create the matrices:
        X = (numVocab, numSentence, numQuestion)
        q = (numQuestion, self.numWord)
        # X = np.zeros((self.numWords, self.maxSentencePerStory, self.numQuestion))
        # q = np.zeros((self.numWords, self.numQuestion))
        """
        X = np.zeros((self.numQuestion, self.maxSentencePerStory, self.numWords))
        q = np.zeros((self.numQuestion, self.numWords))
        a = np.zeros((self.numQuestion, self.numWords))

        currX = np.zeros((self.numWords, self.maxSentencePerStory))
        numQ = 0

        # Create the matrix
        for currFile in self.fileNames:
            fd = open("en/" + currFile, "r")
            for currLine in fd.readlines():
                Id = currLine.split(' ')[0]
                # Start of a new story
                if Id == "1":
                    # Create a new currX
                    currX = np.zeros((self.maxSentencePerStory, self.numWords))
                    # Initialize the story to be at the 0th position
                    numXStory = 0

                # A question
                if '\t' in currLine:
                    # Need to close the current X
                    lineSplit = re.split(r'\t+', currLine)
                    # Don't include the index
                    question = lineSplit[0].split(' ', 1)[1].strip()
                    question = question.translate(None, string.punctuation)
                    answer = lineSplit[1].strip()

                    questionVec = self.getSentenceVector(question)
                    # TODO: What to do with answerVec
                    answerVec = self.getSentenceVector(answer)

                    # Append current X into X
                    X[numQ] = currX

                    # Append q into currentQ
                    q[numQ] = questionVec
                    a[numQ] = answerVec
                    numQ += 1
                # A Sentence
                else:
                    sentence = currLine.split(' ', 1)[1].strip()
                    sentence = sentence.translate(None, string.punctuation)
                    sentenceVec = self.getSentenceVector(sentence)
                    currX[numXStory] = sentenceVec
                    numXStory += 1
        return X, q, a

if __name__=="__main__":
    B = BabiParser()
    sys.exit(0)
    """
    VOCABULARY_SIZE = 50000 #Number of recognized words; V in paper
    MIN_WORD_FREQEUNCY = 5

    #TODO: BATCH/CLEANING

    #Graph parameters
    embed_dim = 128 #Embedding vector dimension; d in paper
    batch_size = 128

    #Create tensorflow graph
    graph = tf.Graph()

    # Run using CPU 
    with graph.as_default(), tf.device('/cpu:0'):
        #Initial loss
        loss = 0.0
        
        story_data = tf.placeholder(tf.int32, shape=[VOCABULARY_SIZE, None])
        question_data = tf.placeholder(tf.int32, shape=[VOCABULARY_SIZE])
        answer_data = tf.placeholder(tf.int32, shape=[VOCABULARY_SIZE]) #1hot vector of answer

        #word encodings
        A_weights = tf.Variable(tf.truncated_normal([VOCABULARY_SIZE, embed_dim], stddev=1.0 / math.sqrt(embed_dim)))
        B_weights = tf.Variable(tf.truncated_normal([VOCABULARY_SIZE, embed_dim], stddev=1.0 / math.sqrt(embed_dim)))
        C_weights = tf.Variable(tf.truncated_normal([VOCABULARY_SIZE, embed_dim], stddev=1.0 / math.sqrt(embed_dim)))
        
        A_biases = tf.Variable(tf.zeros([VOCABULARY_SIZE]))
        B_biases = tf.Variable(tf.zeros([VOCABULARY_SIZE]))
        C_biases = tf.Variable(tf.zeros([VOCABULARY_SIZE]))

        #Prediction weight matrix
        W_weights = tf.Variable(tf.truncated_normal([embed_dim, VOCABULARY_SIZE], stddev=1.0 / math.sqrt(embed_dim)))
        W_biases = tf.Variable(tf.zeros([VOCABULARY_SIZE]))

        #Initialize random embeddings
        embeddings_A = tf.Variable(tf.random_uniform([VOCABULARY_SIZE, embed_dim], -1,1))
        embeddings_B = tf.Variable(tf.random_uniform([VOCABULARY_SIZE, embed_dim], -1,1))
        embeddings_C = tf.Variable(tf.random_uniform([VOCABULARY_SIZE, embed_dim], -1,1))

        #Hidden layers for word encodings
        word_encoder_A = tf.nn.embedding_lookup(embeddings_A, story_data)
        word_encoder_B = tf.nn.embedding_lookup(embeddings_B, question_data)
        word_encoder_C = tf.nn.embedding_lookup(embeddings_C, story_data)
        memory_matrix_m = tf.matmul(story_data, word_encoder_A)
        control_signal_u = tf.matmul(tf.reshape(question_data, [1, VOCABULARY_SIZE]), word_encoder_B)
        c_set = tf.reshape(story_data, [1, VOCABULARY_SIZE]), word_encoder_C)

        memory_selection = tf.matmul(tf.transpose(control_signal_u),memory_matrix_m)
        p = tf.nn.softmax(memory_selection)
        o = tf.reduce_sum(tf.matmul(p, c_set))
        predicted_answer = tf.nn.softmax(tf.matmul(W, tf.sum(o,u)))
        
        #Squared error between predicted and actual answer
        loss += tf.nn.softmax_cross_entropy_with_logits(predicted_answer, answer_data)
        
        #Optimzier
        #TODO: Try using stochastic gradient descent instead
        optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)

        # Fo rplotting loss
        loss_values = []
        #Train the model
        with tf.Session(graph=graph) as session:
            #Initialize variables
            tf.initialize_all_variables().run()
            print('Variables initialized')
            #Restore checkpoint
            if os.path.isfile("memn2n.ckpt"):
                print("Resuming from checkpoint")
                saver.restore(session, "memn2n.ckpt")
            num_steps = 100001
            epoch_size = 1000
            total_epoch_loss =0 
            for step in range(num_steps):
                #TODO Call batch generator and replace train_story, train_qu, train_answer
                train_story = np.zeros([VOCABULARY_SIZE,50])
                train_qu = np.zeros([VOCABULARY_SIZE])
                train_answer = np.zeros([VOCABULARY_SIZE])
                feed_dict = {story_data: train_story, question_data: train qu, answer_data: train_answer}
                _,l = session.run([optimizer, loss], feed_dict = feed_dict)
                total_loss +=l
                if step%50000==0 and step!=0:
                    #Create checkpoint
                    print(t_data)
                    if os.path.isfile("memn2n.ckpt"):
                        os.remove("memn2n.ckpt")
                        checkpoint = saver.save(session, "memn2n.ckpt")
                if step%epoch_size ==0 and step !=0:
                    #Store loss values for the epoch
                    loss_values.append(total_loss/epoch_size)
                    total_loss =0
            embeddings = 
            print("Training done!")
        #Print loss plot
        pylab.ylabel("Loss")
        pylab.xlabel("Step #")
        loss_value_array = np.array(loss_values)
        pylab.plot(np.arange(1,100000, 1001),loss_values)
        pylab.show()  
    """

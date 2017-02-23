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

from matplotlib import pylab #from sklearn.manifold import TSNE

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
    def getBabiTask(self):
        return self.X, self.q, self.a, self.maxSentencePerStory, self.numWords, self.numQuestion

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

    VOCABULARY_SIZE = 19 #Number of recognized words; V in paper
    MIN_WORD_FREQEUNCY = 5
    #TODO change max number of sentences
    max_num_sentences = 10 
    max_sentence_length = 30

    X, q, a, max_num_sentences, VOCABULARY_SIZE, num_steps = B.getBabiTask()

    epoch_size = 2
    print 'epoch size is', epoch_size
    

    #TODO: BATCH/CLEANING

    #Graph parameters
    embed_dim = 80 #Embedding vector dimension; d in paper
    batch_size = 128
    total_loss = 0.0

    #Create tensorflow graph
    graph = tf.Graph()

    with graph.as_default(), tf.device('/cpu:0'):
        #Initial loss
        loss =0
        
        story_data = tf.placeholder(tf.int32, shape=[max_num_sentences, VOCABULARY_SIZE], name="storydata")
        question_data = tf.placeholder(tf.int32, shape=[1,VOCABULARY_SIZE], name="questiondata")
        answer_data = tf.placeholder(tf.int32, shape=[1,VOCABULARY_SIZE], name="answerdata") #1hot vector of answer

        #word encodings
        #A_weights = tf.Variable(tf.truncated_normal([VOCABULARY_SIZE, embed_dim], stddev=1.0 / math.sqrt(embed_dim)))
        #B_weights = tf.Variable(tf.truncated_normal([VOCABULARY_SIZE, embed_dim], stddev=1.0 / math.sqrt(embed_dim)))
        #C_weights = tf.Variable(tf.truncated_normal([VOCABULARY_SIZE, embed_dim], stddev=1.0 / math.sqrt(embed_dim)))
        
        #A_biases = tf.Variable(tf.zeros([VOCABULARY_SIZE,1]))
        #B_biases = tf.Variable(tf.zeros([VOCABULARY_SIZE,1]))
        #C_biases = tf.Variable(tf.zeros([VOCABULARY_SIZE,1]))

        #Prediction weight matrix
        W = tf.Variable(tf.truncated_normal([embed_dim, VOCABULARY_SIZE], stddev=1.0 / math.sqrt(embed_dim)))
        #W_biases = tf.Variable(tf.zeros([embed_dim]))

        #Initialize random embeddings
        embeddings_A = tf.Variable(tf.random_uniform([VOCABULARY_SIZE, embed_dim], -1,1))
        embeddings_B = tf.Variable(tf.random_uniform([VOCABULARY_SIZE, embed_dim], -1,1))
        embeddings_C = tf.Variable(tf.random_uniform([VOCABULARY_SIZE, embed_dim], -1,1))

        #Hidden layers for word encodings (sum words to get sentence representation)
        memory_matrix_m = tf.reduce_sum(tf.nn.embedding_lookup(embeddings_A, story_data),1)
        control_signal_u = tf.reduce_sum(tf.nn.embedding_lookup(embeddings_B, question_data),1)
        c_set= tf.reduce_sum(tf.nn.embedding_lookup(embeddings_C, story_data),1)
        
        memory_selection = tf.matmul(memory_matrix_m, tf.transpose(control_signal_u))
        p = tf.nn.softmax(memory_selection)
        #pdb.set_trace()
        #NOTE: For newer versions of tensorflow, change tf.mul to tf.multiply
        o = tf.reduce_sum(tf.mul(c_set, p),0)

        #Note: For newer versions of tensorflow, change tf.add to tf.sum

        o_u_sum = tf.add(tf.reshape(o, [1,embed_dim]),control_signal_u)
        predicted_answer_labels = tf.nn.softmax(tf.matmul(o_u_sum,W))
        
        #Squared error between predicted and actual answer
        #pdb.set_trace()
        #TODO: Verify that labels should be col vec. and not row vec.
        loss += tf.nn.softmax_cross_entropy_with_logits(predicted_answer_labels, tf.reshape(answer_data, [1,VOCABULARY_SIZE]))
        
        #Optimzier
        #TODO: Try using stochastic gradient descent instead
        optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)



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


            total_epoch_loss =0 

            # Num steps is the number of vocabulary
            for currEpoch in xrange(epoch_size):
                for step in xrange(num_steps):
                    #TODO Call batch generator and replace train_story, train_qu, train_answer
                    '''
                    train_story = np.zeros([10,VOCABULARY_SIZE])
                    train_qu = np.zeros([1,VOCABULARY_SIZE])
                    train_answer = np.zeros([1,VOCABULARY_SIZE])
                    '''

                    train_story = X[step]
                    train_qu = np.reshape(q[step], (1,VOCABULARY_SIZE))
                    train_answer = np.reshape(a[step], (1,VOCABULARY_SIZE))

                    feed_dict = {story_data: train_story, question_data: train_qu, answer_data: train_answer}

                    _,l = session.run([optimizer, loss], feed_dict = feed_dict)

                    total_loss +=l
                    
                    '''
                    if step % 50000==0 and step!=0:
                        #Create checkpoint
                        print(t_data)
                        if os.path.isfile("memn2n.ckpt"):
                            os.remove("memn2n.ckpt")
                            checkpoint = saver.save(session, "memn2n.ckpt")
                    '''
                #Store loss values for the epoch
                loss_values.append(total_loss/num_steps)
                total_loss = 0.0

             
            print("Training done!")

        #Print loss plot
        pylab.ylabel("Loss")
        pylab.xlabel("Step #")
        loss_value_array = np.array(loss_values)
        pylab.plot(np.arange(0,epoch_size, 1),loss_values)
           
        pylab.show()  



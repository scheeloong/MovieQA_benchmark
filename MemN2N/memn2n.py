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
        #self.fileNames = ["qa1_single-supporting-fact_train.txt"]
        self.fileNames = ["qa1_single-supporting-fact_train.txt"]
        #self.fileNames = ["temp.txt"]
        self.vocabularyToIndex = {}
        self.numWords = 1 # Keep 0th index to represent no word
        self.numStory = 0
        self.maxSentencePerStory = 0
        self.numQuestion = 0
        self.maxWordInSentence = 0
        # Parse the vocabulary
        self.parseBabiTaskVocabulary()
        # Parse the vectors to get X, q, a (input, question, answer) sentence vectors
        self.X, self.q, self.a, self.XSentence, self.qSentence, self.aSentence = self.parseBabiTaskVectors()
        print 'numWords:', self.numWords
        print 'numStory:', self.numStory
        print 'maxSentencePerStory:', self.maxSentencePerStory
        print 'maxWordPerSentence:', self.maxWordInSentence
        print 'numQuestion:', self.numQuestion
        print 'Input Shape:', self.X.shape
        print 'Question Shape:', self.q.shape
        print 'Answer Shape:', self.a.shape
        '''
        print self.X
        print self.XSentence
        print self.q
        print self.qSentence
        print self.a 
        print self.aSentence
        '''

    def getBabiTask(self):
        return self.X, self.q, self.a, self.maxSentencePerStory, self.maxWordInSentence, self.numQuestion, self.numWords

    def insertVocabulary(self, word):
        if word in self.vocabularyToIndex:
            return
        self.vocabularyToIndex[word] = self.numWords
        self.numWords += 1
        return

    def getSentenceVector(self, oneHotSentence):
        # Assign default values to a word that doesn't exist in the dictionary
        sentenceVec  = np.zeros(self.maxWordInSentence)
        index = 0
        vocabularyIndex = 0
        for val in oneHotSentence:
            if val == 1:
                sentenceVec[index] = vocabularyIndex
                index += 1
            vocabularyIndex += 1
        return sentenceVec

    def getSentenceOneHotVector(self, sentence):
        rowVector = list()
        for word in sentence.split():
            rowVector.append(self.vocabularyToIndex[word])
        sentenceVec = self.convertToOneHot(np.array(rowVector))
        return sentenceVec

    def convertToOneHot(self, rowVector):
        oneHotVectors = np.eye(self.numWords)[rowVector]
        return np.sum(oneHotVectors, 0)

    def parseBabiTaskVocabulary(self, addQToStory=False):
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
                    self.maxWordInSentence = max(len(question.split(' ')), self.maxWordInSentence)
                    self.maxWordInSentence = max(len(answer.split(' ')), self.maxWordInSentence)
                    for currWord in question.split(' '):
                        self.insertVocabulary(currWord)
                    for currWord in answer.split(' '):
                        self.insertVocabulary(currWord)
                    if addQToStory:
                        numSentencePerStory += 1
                        self.maxSentencePerStory = max(self.maxSentencePerStory, numSentencePerStory)
                # A Sentence
                else:
                    numSentencePerStory += 1
                    self.maxSentencePerStory = max(self.maxSentencePerStory, numSentencePerStory)
                    sentence = currLine.split(' ', 1)[1].strip()
                    sentence = sentence.translate(None, string.punctuation)
                    self.maxWordInSentence = max(len(sentence.split(' ')), self.maxWordInSentence)
                    for currWord in sentence.split(' '):
                        self.insertVocabulary(currWord)
        # Done inserting all word
        print self.numWords

    def parseBabiTaskVectors(self, addQToStory=False):
        X = np.zeros((self.numQuestion, self.maxSentencePerStory, self.numWords))
        q = np.zeros((self.numQuestion, self.numWords))
        a = np.zeros((self.numQuestion, self.numWords))

        XSent = np.zeros((self.numQuestion, self.maxSentencePerStory, self.maxWordInSentence))
        qSent = np.zeros((self.numQuestion, self.maxWordInSentence))
        aSent = np.zeros((self.numQuestion, self.maxWordInSentence))

        currX = np.zeros((self.maxSentencePerStory, self.numWords))
        currXSent = np.zeros((self.maxSentencePerStory, self.maxWordInSentence))
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
                    currXSent = np.zeros((self.maxSentencePerStory, self.maxWordInSentence))
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

                    questionVec = self.getSentenceOneHotVector(question)
                    answerVec = self.getSentenceOneHotVector(answer)
                    questionVecSent = self.getSentenceVector(questionVec)
                    answerVecSent = self.getSentenceVector(answerVec)

                    if addQToStory:
                        currX[numXStory] = questionVec
                        currXSent[numXStory] = questionVecSent
                        numXStory += 1
                    # Append current X into X
                    X[numQ] = currX
                    XSent[numQ] = currXSent

                    # Append q into currentQ
                    q[numQ] = questionVec
                    qSent[numQ] = questionVecSent
                    a[numQ] = answerVec
                    aSent[numQ] = answerVecSent
                    numQ += 1

                # A Sentence
                else:
                    sentence = currLine.split(' ', 1)[1].strip()
                    sentence = sentence.translate(None, string.punctuation)
                    sentenceVec = self.getSentenceOneHotVector(sentence)
                    sentenceVecSent = self.getSentenceVector(sentenceVec)
                    currX[numXStory] = sentenceVec
                    currXSent[numXStory] = sentenceVecSent 
                    numXStory += 1
        return X, q, a, XSent, qSent, aSent

def ShuffleBatches(trainData, trainDataTwo, trainTarget):
    # Gets the state as the current time
    rngState = np.random.get_state()
    np.random.shuffle(trainData)
    np.random.set_state(rngState)
    np.random.shuffle(trainDataTwo)
    np.random.set_state(rngState)
    np.random.shuffle(trainTarget)
    return trainData, trainDataTwo, trainTarget

if __name__=="__main__":
    B = BabiParser()
    MIN_WORD_FREQEUNCY = 5
    X, q, a, max_num_sentences, SENTENCE_LENGTH, num_steps, VOCABULARY_SIZE = B.getBabiTask()
    valStart = (num_steps*8)/10
    testStart = (num_steps*9)/10
    valX = X[valStart:testStart]
    valq = q[valStart:testStart]
    vala = a[valStart:testStart]
    # Only 10% taken our for validation according to 4.1
    valX = X[testStart:]
    valq = q[testStart:]
    vala = a[testStart:]
    '''
    testx = x[teststart:]
    testq = q[teststart:]
    testa = a[teststart:]
    '''
    '''
    X = X[:valStart]
    q = q[:valStart]
    a = a[:valStart]
    num_steps = valStart
    # TODO:
    # '''
    X = X[:testStart]
    q = q[:testStart]
    a = a[:testStart]
    num_steps = testStart

    epoch_size = 100
    print 'epoch size is', epoch_size

    #Graph parameters
    num_hops = 3 # TODO Implement number of hops
    embed_dim = 20 # Embedding vector dimension; d in paper for independent training
    batch_size = 32
    total_loss = 0.0
    learningRate = 0.02 # according to paper, will be 0.01 before first iteration
    #Create tensorflow graph
    graph = tf.Graph()

    with graph.as_default(), tf.device('/cpu:0'):
        story_data = tf.placeholder(tf.float32, shape=[None, max_num_sentences, VOCABULARY_SIZE], name="storydata")
        question_data = tf.placeholder(tf.float32, shape=[None, 1, VOCABULARY_SIZE], name="questiondata")
        answer_data = tf.placeholder(tf.float32, shape=[None, 1, VOCABULARY_SIZE], name="answerdata") #1hot vector of answer
        #Prediction weight matrix
        W = tf.Variable(tf.truncated_normal([embed_dim, VOCABULARY_SIZE], stddev=0.1)) # 5.1 of paper
        W_biases = tf.Variable(tf.truncated_normal([VOCABULARY_SIZE], stddev=0.1))

        #Initialize random embeddings
        # Initialize as normal distribution with mean = 0 and std.deviation = 1 according to paper
        # To perform matrix multiplication on higher dimensions
        batchSizing= tf.shape(story_data)[0]
        embeddings_A = tf.Variable(tf.truncated_normal([VOCABULARY_SIZE, embed_dim], stddev=0.05), name="VariableEmbeddingA", dtype=tf.float32)
        embeddings_B = tf.Variable(tf.truncated_normal([VOCABULARY_SIZE, embed_dim], stddev=0.05), name="VariableEmbeddingB")
        embeddings_C = tf.Variable(tf.truncated_normal([VOCABULARY_SIZE, embed_dim], stddev=0.05), name="VariableEmbeddingC")
        '''
        -------------------------------------------------------
        b = batch_size = 9
        V = vocabulary size = 20
        z = numberOfSentence = 10
        l = numberOfWordsInSentence = 6
        d = word_embedding_size = 2
        -------------------------------------------------------
        W  = (d, V) 
        story data (9, 10, 6) = (b, z, l)
        Embeddings A (20, 2) = (V, d)
        Embedding Lookup A (9, 10, 6, 2) = (b, z, l, d)
        Memory Matrix M (9, 10, 2) = (b, z, d)
        Question Data (9, 1, 6) = (b, 1, l)
        Control Signal U (9, 1, 2) = (b, 1, d)
        Memory Selection (9, 10, 1) = (b, z, 1)
        p (9, 10, 1) = (b, z, 1)
        c_set (9, 2, 10) = (b, d, z)
        o (9, 2) = (b, d, 1)
        o_u_sum = (b, 1, d)
        Correct Prediction (9, 1) = (b, 1)
        -------------------------------------------------------
        '''
        memory_matrix_m = tf.reshape(tf.matmul(tf.reshape(story_data, (batchSizing*max_num_sentences,VOCABULARY_SIZE)), embeddings_A), (batchSizing, max_num_sentences, embed_dim))
        haha = memory_matrix_m  # TODO REMOVE FROM DEBUGGING
        # Hidden layers for word encodings (sum words to get sentence representation)
        # This gets a sentence representation for each sentence in a paragraph
        # Gets a single sentence representation for that 1 question
        control_signal_u= tf.matmul(tf.reshape(question_data, (batchSizing, VOCABULARY_SIZE)), embeddings_B)
        # Get training control values
        c_set = tf.reshape(tf.matmul(tf.reshape(story_data, (batchSizing*max_num_sentences,VOCABULARY_SIZE)), embeddings_C), (batchSizing, max_num_sentences, embed_dim))
        # Use memory multplied with control to select a story
        # (b,z,d) * (b,d,1) = (b,z,1)
        memory_selection = tf.reshape(tf.matmul(memory_matrix_m, tf.reshape(control_signal_u, (batchSizing, embed_dim, 1))), (batchSizing, max_num_sentences, 1))
        # Calculate which story to select
        p = tf.nn.softmax(memory_selection, 1)
        # Select the story
        c_set = tf.transpose(c_set, (0, 2, 1))
        o = tf.reshape(tf.matmul(c_set, tf.reshape(p,(batchSizing, max_num_sentences, 1))),(batchSizing, embed_dim))
        # Calculate the sum
        o_u_sum = tf.add(o, control_signal_u)
        # predicted_answer_labels = tf.matmul(o_u_sum, W) + W_biases 
        predicted_answer_labels = tf.matmul(o_u_sum, W)
        predicted_answer_labels = tf.reshape(predicted_answer_labels, [-1, 1, VOCABULARY_SIZE])
        y_predicted = predicted_answer_labels
        answer_data = tf.reshape(answer_data, [batchSizing, 1, VOCABULARY_SIZE])
        y_target = answer_data
        # Multi-class Classification
        argyPredict  = tf.argmax(y_predicted,2)
        argyTarget = tf.argmax(y_target,2)
        correctPred = tf.equal(argyPredict, argyTarget)
        accuracy = tf.reduce_mean(tf.cast(correctPred, "float"))
        # Paper said it didn't average the loss, but it will reach infinity if batch size is too large
        loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits = y_predicted, labels = y_target))
        #Optimizer
        optimizer = tf.train.AdagradOptimizer(learningRate).minimize(loss)
        # '''
        '''
        optimizers = tf.train.GradientDescentOptimizer(learningRate)
        grads_and_vars = optimizers.compute_gradients(loss)
        grads_and_vars = [(tf.clip_by_norm(g, 40.0), v) for g, v in grads_and_vars]
        optimizer = optimizers.apply_gradients(grads_and_vars)
        # '''
        #optimizer = tf.train.AdamOptimizer(learningRate).minimize(loss)
        #optimizer = tf.train.GradientDescentOptimizer(learningRate).minimize(loss)

        loss_values = []
        #Train the model
        with tf.Session(graph=graph) as session:
            #Initialize variables
            tf.global_variables_initializer().run()
            val_story = valX[:]
            #val_qu = np.reshape(valq[:], (valq.shape[0],1, SENTENCE_LENGTH))
            val_qu = np.reshape(valq[:], (valq.shape[0],1, VOCABULARY_SIZE))
            val_a = np.reshape(vala[:], (vala.shape[0],1, VOCABULARY_SIZE))
            feed_dictV = {story_data: val_story, question_data: val_qu, answer_data: val_a}
            '''
            test_story = testX[:]
            test_qu = np.reshape(testq[:], (testq.shape[0],1, SENTENCE_LENGTH))
            test_a = np.reshape(testa[:], (testa.shape[0],1, VOCABULARY_SIZE))
            feed_dictT = {story_data: test_story, question_data: test_qu, answer_data: test_a}
            '''
            total_loss = 0.0
            # Num steps is the total number of questions
            for currEpoch in xrange(epoch_size):
                X, q, a = ShuffleBatches(X,q,a) 
                numCorrect = 0.0
                for step in xrange(num_steps/batch_size):
                    train_story = X[step*batch_size:(step+1)*batch_size]
                    #train_qu = np.reshape(q[step*batch_size:(step+1)*batch_size], (batch_size,1,SENTENCE_LENGTH))
                    train_qu = np.reshape(q[step*batch_size:(step+1)*batch_size], (batch_size,1, VOCABULARY_SIZE))
                    train_answer = np.reshape(a[step*batch_size:(step+1)*batch_size], (batch_size, 1,VOCABULARY_SIZE))
                    feed_dictS = {story_data: train_story, question_data: train_qu, answer_data: train_answer}

                    _,l,yhat,y, acc, argyhat, argy, correctPrediction = session.run([optimizer, loss, predicted_answer_labels, answer_data, accuracy, argyPredict, argyTarget, correctPred], feed_dict = feed_dictS)

                    '''
                    a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13 = session.run([story_data, embeddings_A, haha, memory_matrix_m, question_data, control_signal_u, memory_selection, p, c_set, o, o_u_sum, correctPred, accuracy], feed_dict = feed_dictS)
                    print "story data", a1.shape, a1 
                    print "Embeddings A", a2.shape, a2
                    print "Embedding Lookup A", a3.shape, a3
                    print "Memory Matrix M", a4.shape, a4
                    print "Question Data", a5.shape, a5
                    print "Control Signal U" , a6.shape, a6
                    print "Memory Selection", a7.shape, a7
                    print "p", a8.shape, a8
                    print "c_set", a9.shape, a9
                    print "o", a10.shape, a10
                    print "o_u_sum ", a11.shape, a11
                    print "Correct Prediction", a12.shape, a12
                    print "Accuracy", a13.shape, a13
                    print "story data", a1.shape
                    print "Embeddings A", a2.shape
                    print "Embedding Lookup A", a3.shape
                    print "Memory Matrix M", a4.shape
                    print "Question Data", a5.shape
                    print "Control Signal U" , a6.shape
                    print "Memory Selection", a7.shape
                    print "p", a8.shape
                    print "c_set", a9.shape
                    print "o", a10.shape
                    print "o_u_sum ", a11.shape
                    print "Correct Prediction", a12.shape
                    print "Accuracy", a13.shape
                    sys.exit(0)
                    # '''

                    '''
                    print 'EVALUATION YHAT AND Y'
                    print 'yhat', yhat
                    print 'argyhat', argyhat
                    print 'y', y
                    print 'argy', argy
                    print l
                    '''
                    #numCorrect += acc # DOesnt work for batchsize > 1
                    #print "CorrectPrediction", correctPrediction
                    total_loss += l
                    numCorrect += sum(correctPrediction)
                #Store loss values for the epoch
                loss_values.append(total_loss)
                accuracyThisEpoch = numCorrect/float(num_steps)
                valLoss, valAccuracy = session.run([loss, accuracy], feed_dict = feed_dictV)
                #testLoss, testAccuracy = session.run([loss, accuracy], feed_dict = feed_dictT)
                print "valLoss", valLoss
                print "valAcc", valAccuracy
                #print "testLoss", testLoss
                #print "testAcc", testAccuracy
                print 'EpochNum:', currEpoch
                print 'LearningRate:', learningRate
                print 'TotalLossCurrEpoch:', total_loss
                print 'AccuracyCurrEpoch:', accuracyThisEpoch
                total_loss = 0.0
                numCorrect = 0.0
                #if not currEpoch % 25:
                if not currEpoch % 15:
                    # LearningRate Annealing
                    learningRate = learningRate/2.0 # 4.1 Annealing. 
             
            print("Training done!")

        #Print loss plot
        pylab.ylabel("Loss")
        pylab.xlabel("Step #")
        loss_value_array = np.array(loss_values)
        pylab.plot(np.arange(0,epoch_size, 1),loss_values)
           
        pylab.show()  

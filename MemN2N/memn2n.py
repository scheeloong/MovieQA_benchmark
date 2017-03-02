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
        return self.XSentence, self.qSentence, self.a, self.maxSentencePerStory, self.maxWordInSentence, self.numQuestion, self.numWords

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
                    self.maxWordInSentence = max(len(question.split(' ')), self.maxWordInSentence)
                    self.maxWordInSentence = max(len(answer.split(' ')), self.maxWordInSentence)
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
                    self.maxWordInSentence = max(len(sentence.split(' ')), self.maxWordInSentence)
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

if __name__=="__main__":
    B = BabiParser()
    MIN_WORD_FREQEUNCY = 5
    X, q, a, max_num_sentences, SENTENCE_LENGTH, num_steps, VOCABULARY_SIZE = B.getBabiTask()
    epoch_size = 100 # 25 * 4 
    print 'epoch size is', epoch_size

    #TODO: BATCH/CLEANING
    #Graph parameters
    num_hops = 3 # TODO Implement number of hops
    embed_dim = 20 # Embedding vector dimension; d in paper for independent training
    #embed_dim = 2 # TEMP DEBUG
    batch_size = 128 # TODO: Implement this
    total_loss = 0.0
    learningRate = 0.02 # according to paper, will be 0.01 before first iteration
    #Create tensorflow graph
    graph = tf.Graph()

    with graph.as_default(), tf.device('/cpu:0'):
        story_data = tf.placeholder(tf.int32, shape=[max_num_sentences, SENTENCE_LENGTH], name="storydata")
        question_data = tf.placeholder(tf.int32, shape=[1,SENTENCE_LENGTH], name="questiondata")
        answer_data = tf.placeholder(tf.int32, shape=[1,VOCABULARY_SIZE], name="answerdata") #1hot vector of answer

        #word encodings
        #A_weights = tf.Variable(tf.truncated_normal([VOCABULARY_SIZE, embed_dim], stddev=1.0 / math.sqrt(embed_dim)))
        #B_weights = tf.Variable(tf.truncated_normal([VOCABULARY_SIZE, embed_dim], stddev=1.0 / math.sqrt(embed_dim)))
        #C_weights = tf.Variable(tf.truncated_normal([VOCABULARY_SIZE, embed_dim], stddev=1.0 / math.sqrt(embed_dim)))
        
        #A_biases = tf.Variable(tf.zeros([VOCABULARY_SIZE,1]))
        #B_biases = tf.Variable(tf.zeros([VOCABULARY_SIZE,1]))
        #C_biases = tf.Variable(tf.zeros([VOCABULARY_SIZE,1]))

        #Prediction weight matrix
        W = tf.Variable(tf.truncated_normal([embed_dim, VOCABULARY_SIZE], stddev=0.05)) # 5.1 of paper
        #W_biases = tf.Variable(tf.zeros([embed_dim]))


        #Initialize random embeddings
        # Initialize as normal distribution with mean = 0 and std.deviation = 1 according to paper
        embeddings_A_temp = tf.Variable(tf.truncated_normal([VOCABULARY_SIZE-1, embed_dim], stddev=0.1), name="VariableEmbeddingA", dtype=tf.float32)
        embeddings_B_temp = tf.Variable(tf.truncated_normal([VOCABULARY_SIZE-1, embed_dim], stddev=0.1), name="VariableEmbeddingB")
        embeddings_C_temp = tf.Variable(tf.truncated_normal([VOCABULARY_SIZE-1, embed_dim], stddev=0.1), name="VariableEmbeddingC")
        # Append 0 values for no words
        noWordA = tf.Variable(tf.zeros([1, embed_dim]), dtype=tf.float32)
        noWordB = tf.Variable(tf.zeros([1, embed_dim]), dtype=tf.float32)
        noWordC = tf.Variable(tf.zeros([1, embed_dim]), dtype=tf.float32)
        embeddings_A = tf.concat([embeddings_A_temp, noWordA], 0)
        embeddings_B = tf.concat([embeddings_B_temp, noWordB], 0)
        embeddings_C = tf.concat([embeddings_C_temp, noWordC], 0)



        haha = tf.nn.embedding_lookup(embeddings_A, story_data, name="EmbeddingM")
        #Hidden layers for word encodings (sum words to get sentence representation)
        # This gets a sentence representation for each sentence in a paragraph
        memory_matrix_m = tf.reduce_sum(tf.nn.embedding_lookup(embeddings_A, story_data, name="EmbeddingM"),1)

        # Gets a single sentence representation for that 1 question
        control_signal_u = tf.reduce_sum(tf.nn.embedding_lookup(embeddings_B, question_data, name="EmbeddingU"),1)

        # Get training control values
        c_set= tf.reduce_sum(tf.nn.embedding_lookup(embeddings_C, story_data, name="EmbeddingCSet"),1)
       
        # Use memory multplied with control to select a story
        memory_selection = tf.matmul(memory_matrix_m, tf.transpose(control_signal_u))
        #memory_selection = tf.matmul(memory_matrix_m, control_signal_u)

        # Calculate which story to select
        p = tf.nn.softmax(memory_selection, 0)
        #pdb.set_trace()

        # Select the story
        o = tf.reduce_sum(tf.multiply(c_set, p), 0)

        #Note: For newer versions of tensorflow, change tf.add to tf.sum

        # Calculate the sum
        o_u_sum = tf.add(tf.reshape(o, [1,embed_dim]),control_signal_u)

        # Get the sum multiplied by W
        predicted_answer_labels = tf.matmul(o_u_sum,W)
        y_predicted = predicted_answer_labels

        #Squared error between predicted and actual answer
        #pdb.set_trace()
        #TODO: Verify that labels should be col vec. and not row vec.
        answerY = tf.reshape(answer_data, [1,VOCABULARY_SIZE])
        y_target = answerY

        # Multi-class Classification
        argyPredict  = tf.argmax(y_predicted,1)
        argyTarget = tf.argmax(y_target,1)
        correctPred = tf.equal(tf.argmax(y_predicted, 1), tf.argmax(y_target, 1))
        accuracy = tf.cast(correctPred, "float")

        loss = tf.nn.softmax_cross_entropy_with_logits(logits = predicted_answer_labels, labels = tf.reshape(answer_data, [1,VOCABULARY_SIZE]))
        
        #Optimizer
        #optimizer = tf.train.AdagradOptimizer(learningRate).minimize(loss)
        #optimizer = tf.train.AdamOptimizer(learningRate).minimize(loss)
        optimizer = tf.train.GradientDescentOptimizer(learningRate).minimize(loss)

        loss_values = []
        
        #Train the model
        with tf.Session(graph=graph) as session:
            #Initialize variables
            tf.global_variables_initializer().run()
            print('Variables initialized')

            #Restore checkpoint
            if os.path.isfile("memn2n.ckpt"):
                print("Resuming from checkpoint")
                saver.restore(session, "memn2n.ckpt")

            total_loss = 0.0
            # Num steps is the total number of questions
            for currEpoch in xrange(epoch_size):
                numCorrect = 0.0
                for step in xrange(num_steps):
                    #TODO Call batch generator and replace train_story, train_qu, train_answer
                    '''
                    train_story = np.zeros([10,VOCABULARY_SIZE])
                    train_qu = np.zeros([1,VOCABULARY_SIZE])
                    train_answer = np.zeros([1,VOCABULARY_SIZE])
                    '''

                    train_story = X[step]
                    train_qu = np.reshape(q[step], (1,SENTENCE_LENGTH))
                    train_answer = np.reshape(a[step], (1,VOCABULARY_SIZE))

                    feed_dictS = {story_data: train_story, question_data: train_qu, answer_data: train_answer}

                    # TODO: Uncomment once done
                    _,l,yhat,y, acc, argyhat, argy = session.run([optimizer, loss, predicted_answer_labels, answerY, accuracy, argyPredict, argyTarget], feed_dict = feed_dictS)
                    '''
                    _, a1, a2, a3, a4, a5, a6, a7, a8, a9= session.run([optimizer, story_data, embeddings_A, haha, memory_matrix_m, question_data, control_signal_u, memory_selection, p, o], feed_dict = feed_dictS)
                    print "story data", a1 
                    print "Embeddings A", a2
                    print "Embedding Lookup A", a3
                    print "Memory Matrix M", a4
                    print "Question Data", a5
                    print "Control Signal U" , a6
                    print "Memory Selection", a7
                    print "p", a8
                    print "o", a9
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
                    
                    numCorrect += acc
                    total_loss += l

                    
                    '''
                    if step % 50000==0 and step!=0:
                        #Create checkpoint
                        print(t_data)
                        if os.path.isfile("memn2n.ckpt"):
                            os.remove("memn2n.ckpt")
                            checkpoint = saver.save(session, "memn2n.ckpt")
                    '''
                #Store loss values for the epoch
                loss_values.append(total_loss)
                accuracyThisEpoch = numCorrect/float(num_steps)
                print 'EpochNum:', currEpoch
                print 'LearningRate:', learningRate
                print 'TotalLossCurrEpoch:', total_loss
                print 'Accuracy:', accuracyThisEpoch
                total_loss = 0.0
                numCorrect = 0.0
                if not currEpoch % 25:
                    # LearningRate Annealing
                    learningRate = learningRate/2.0
             
            print("Training done!")

        #Print loss plot
        pylab.ylabel("Loss")
        pylab.xlabel("Step #")
        loss_value_array = np.array(loss_values)
        pylab.plot(np.arange(0,epoch_size, 1),loss_values)
           
        pylab.show()  



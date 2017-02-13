import os
import math
import random
import string
import zipfile
import collections
import random
import pdb

import tensorflow as tf
import numpy as np

from matplotlib import pylab
#from sklearn.manifold import TSNE

from collections import OrderedDict, defaultdict, Counter, deque
from random import shuffle

if __name__=="__main__":
    VOCABULARY_SIZE = 50000 #Number of recognized words; V in paper
    MIN_WORD_FREQEUNCY = 5

    

    #TODO: BATCH/CLEANING

    #Graph parameters
    embed_dim = 80 #Embedding vector dimension; d in paper
    batch_size = 128


    #Create tensorflow graph
    graph = tf.Graph()

    with graph.as_default(), tf.device('/cpu:0'):
        #Initial loss
        loss =0
        
        story_data = tf.placeholder(tf.int32, shape=[None, VOCABULARY_SIZE])
        question_data = tf.placeholder(tf.int32, shape=[VOCABULARY_SIZE, 1])
        answer_data = tf.placeholder(tf.int32, shape=[VOCABULARY_SIZE, 1]) #1hot vector of answer

        #word encodings
        A_weights = tf.Variable(tf.truncated_normal([VOCABULARY_SIZE, embed_dim], stddev=1.0 / math.sqrt(embed_dim)))
        B_weights = tf.Variable(tf.truncated_normal([VOCABULARY_SIZE, embed_dim], stddev=1.0 / math.sqrt(embed_dim)))
        C_weights = tf.Variable(tf.truncated_normal([VOCABULARY_SIZE, embed_dim], stddev=1.0 / math.sqrt(embed_dim)))
        
        #A_biases = tf.Variable(tf.zeros([VOCABULARY_SIZE,1]))
        #B_biases = tf.Variable(tf.zeros([VOCABULARY_SIZE,1]))
        #C_biases = tf.Variable(tf.zeros([VOCABULARY_SIZE,1]))

        #Prediction weight matrix
        W_weights = tf.Variable(tf.truncated_normal([embed_dim, VOCABULARY_SIZE], stddev=1.0 / math.sqrt(embed_dim)))
        W_biases = tf.Variable(tf.zeros([embed_dim]))

        #Initialize random embeddings
        embeddings_A = tf.Variable(tf.random_uniform([VOCABULARY_SIZE, embed_dim], -1,1))
        embeddings_B = tf.Variable(tf.random_uniform([VOCABULARY_SIZE, embed_dim], -1,1))
        embeddings_C = tf.Variable(tf.random_uniform([VOCABULARY_SIZE, embed_dim], -1,1))

        #Hidden layers for word encodings
        memory_matrix_m = tf.nn.embedding_lookup(embeddings_A, story_data)
        control_signal_u = tf.nn.embedding_lookup(embeddings_B, question_data)
        c_set= tf.nn.embedding_lookup(embeddings_C, story_data)
        pdb.set_trace()

        memory_selection = tf.matmul(memory_matrix_m, tf.transpose(control_signal_u))
        p = tf.nn.softmax(memory_selection)
        
        
        o = tf.reduce_sum(tf.matmul(c_set, p))

        predicted_answer = tf.nn.softmax(tf.matmul(W, tf.sum(o,u)))
        
        #Squared error between predicted and actual answer
        loss += tf.nn.softmax_cross_entropy_with_logits(predicted_answer, answer_data)
        
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


            num_steps = 100001
            epoch_size = 1000
            total_epoch_loss =0 

            for step in range(num_steps):
                #TODO Call batch generator and replace train_story, train_qu, train_answer
                train_story = np.zeros([VOCABULARY_SIZE,50])
                train_qu = np.zeros([VOCABULARY_SIZE])
                train_answer = np.zeros([VOCABULARY_SIZE])

                feed_dict = {story_data: train_story, question_data: train_qu, answer_data: train_answer}

                
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

             
            print("Training done!")

        #Print loss plot
        pylab.ylabel("Loss")
        pylab.xlabel("Step #")
        loss_value_array = np.array(loss_values)
        pylab.plot(np.arange(1,100000, 1001),loss_values)
           
        pylab.show()  



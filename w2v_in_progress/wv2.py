##word2vec implementation with no shared decoder
from __future__ import print_function
import os
import math
import random
import string
import zipfile
import collections
import random
import zipfile

import tensorflow as tf
import numpy as np

from matplotlib import pylab
#from sklearn.manifold import TSNE

from collections import OrderedDict, defaultdict, Counter, deque
from random import shuffle

#----------------------------------------------PREPROCESSING-----------------------------------------

def load_MovieQA_files():
    '''Load all MovieQA files and prepares for preprocessing'''

    path = 'MovieQA_benchmark-master/story/plot'
    parsed_movie_dict = {}
    file_list= os.listdir(path)
    all_MovieQA_words = []

    for file_i in file_list:
        filename = file_i
        with open(path + '/' + filename) as file:
            text = file.read()

            parsed_file = word_parse(text)

            parsed_movie_dict[filename] = parsed_file

            #Add to list of all movie text
            all_MovieQA_words.extend(parsed_file)
    
    return (parsed_movie_dict, all_MovieQA_words)

def word_parse(file_text):
    """given a file, remove all punctuation/capitalization and 
    place all words into a list.
    """
    word_list = []

    file_text = file_text.lower()
    file_text = file_text.replace('\n', ' ')

    skip_chars = ',:;()!?.'
    for char in skip_chars:
        file_text = file_text.replace(char, ' ')
    word_list = file_text.split()
    
    return word_list

def add_word_to_vocabulary(movie_to_words_map, vocabulary_counter):
    """Adds words to vocabulary_counter, updates word frequency"""
    
    for movie in movie_to_words_map:
        for word in movie_to_words_map[movie]:
            if word in vocabulary_counter:
                vocabulary_counter[word] +=1
            else: 
                vocabulary_counter[word] =1

    return vocabulary_counter

def map_words_to_one_hot_indices(recognized_words, vocabulary_counter):
    """Assigns a one-hot index (position in 1-hot vector) to each recognized word"""
    one_hot_index = 1
    one_hot_indices_map = {}
    
    for _, word in enumerate(vocabulary_counter):
        if word in recognized_words:
            one_hot_indices_map[word] = one_hot_index
            one_hot_index +=1
        else:
            #TODO: remove else statement
            one_hot_indices_map[word] = 0

    return one_hot_indices_map
    

#----------------------------------------BATCH GENERATION----------------------------------------
 
def split_data(window_size, movieQA_data):
    """Split data into training, validation, and test sets"""
    movieQA_num_words = len(movieQA_data)
    center_word_indices = list(range(window_size, movieQA_num_words - window_size+1)) #Clip off ends to avoid going out of range
    num_valid_example_words = len(center_word_indices)
    
    #Split (Important: only shuffle training set after this split)
    train_set_size = int(math.floor(num_valid_example_words*0.7))
    test_set_size = int(math.floor(num_valid_example_words*0.2))
    valid_set_size = num_valid_example_words - (train_set_size+test_set_size)

    train_set_center_words = center_word_indices[:train_set_size]
    test_set_center_words = center_word_indices[train_set_size: (train_set_size + test_set_size)]
    valid_set_center_words = center_word_indices[(train_set_size+test_set_size):] 

    shuffle(train_set_center_words)

    return (train_set_center_words, test_set_center_words, valid_set_center_words)

def generate_batch(num_skips, batch_size, movieQA_data, start_index, center_word_indices):
    '''Generate a training batch'''
    #Data is the one-hot index representation of the words in the vocabulary
    #num_skips is number of skips to do from each center word
    #start index is the position in ex_indices_array 

    assert batch_size%num_skips == 0 #Ensure batch size is divisible by number of skips
    assert num_skips%2 ==0

    num_center_words = batch_size
    num_side_words = num_skips/2

    batch_data = np.ndarray(shape=(batch_size), dtype=np.int32) #Holds the center words of current training batch
    labels = np.ndarray(shape=(batch_size,num_skips), dtype=np.int32) #holds the labels of the current training batch
    
    for i in range(num_center_words):
        #Fill training batch/labels
        center_word_index = center_word_indices[i]
        prev_context = movieQA_data[center_word_index-num_side_words:center_word_index]
        following_context = movieQA_data[center_word_index+1:center_word_index+num_side_words+1]
        context_to_predict = prev_context+following_context

        batch_data[i] = movieQA_data[center_word_index]
        for context_word in range(num_skips):
            labels[i,context_word] = context_to_predict[context_word]
        
   
    return(batch_data,labels)



#----------------------------------------TRAIN THE MODEL-----------------------------------------  

if __name__=="__main__":
    
    VOCABULARY_SIZE = 10000
    MIN_WORD_FREQUENCY = 5 #Minimum frequency for a word to be recognized
    
    vocabulary_counter = Counter()   
    
    (movie_to_words_map, all_MovieQA_words) = load_MovieQA_files()
    vocabulary_counter = add_word_to_vocabulary(movie_to_words_map, vocabulary_counter)

    recognized_words_counter = vocabulary_counter.most_common(VOCABULARY_SIZE)
    recognized_words = []

    #Take VOCABULARY_SIZE most common words
    for word, _ in recognized_words_counter:
        recognized_words.append(word)

    #Get one-hot-index mappings for recognized words
    one_hot_indices_map = map_words_to_one_hot_indices(recognized_words, vocabulary_counter)   
    
    #Get one-hot-index representation of movieQA words
    movieQA_data =[]
    for word in all_MovieQA_words:
        if word in one_hot_indices_map:
            movieQA_data.append(one_hot_indices_map[word])
        else:
            movieQA_data.append(0) #Unrecognized word


    #Graph parameters
    window_size = 4 #Num context words on either side of center word
    num_skips = window_size*2
    embed_dim = 128 #Embedding vector dimension
    batch_size = 128
    start_index = 0
    num_sampled = 64 #Negative examples

    #Split dataset
    train_set_center_words, test_set_center_words, valid_set_center_words = split_data(window_size, movieQA_data)

    #Initialize tensorflow graph
    graph = tf.Graph()
    
    with graph.as_default(),tf.device('/cpu:0'):
        #Initial loss
        loss =0
 
        #The training data (center words) and training labels (predictions)
        train_batch_data = tf.placeholder(tf.int32, shape=[batch_size])
        train_batch_labels = tf.placeholder(tf.int32, shape=[batch_size,num_skips])

        #Initialize random embeddings
        embeddings = tf.Variable(tf.random_uniform([VOCABULARY_SIZE, embed_dim], -1,1))
        
        #Hidden layer 1
        conn = tf.nn.embedding_lookup(embeddings, train_batch_data)

        #Weights/biases for each context prediction
        for i in range(num_skips):
            weights = tf.Variable(tf.truncated_normal([VOCABULARY_SIZE, embed_dim], stddev=1.0 / math.sqrt(embed_dim)))
            biases = tf.Variable(tf.zeros([VOCABULARY_SIZE]))
            #Accumulate loss across all context words
            #TODO: Consider averaging loss
            loss += tf.reduce_mean(tf.nn.sampled_softmax_loss(weights, biases, conn, train_batch_labels[:,i:i+1], num_sampled, VOCABULARY_SIZE))
  
        #Gradient Descent optimizer
        optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)

        #Cosine Similarity
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings),1, keep_dims=True))
        normalized_embeddings = embeddings/norm

        #To plot loss
        loss_values = []
        
        #Saver to backup training progress
        saver = tf.train.Saver()
        
        #Train the model
        with tf.Session(graph=graph) as session:
            #Initialize variables
            tf.initialize_all_variables().run()
            print('Variables initialized')

            #Restore checkpoint
            if os.path.isfile("wv2.ckpt"):
                print("Resuming from checkpoint")
                saver.restore(session, "wv2.ckpt")

            num_steps = 100001
            epoch_size = 1000
            start_index =0 #iterate through the center indices
            total_loss = 0 #Loss for the epoch

            for step in range(num_steps):
                t_data, t_labels = generate_batch(num_skips, batch_size, movieQA_data, start_index, train_set_center_words)
                feed_dict = {train_batch_data: t_data, train_batch_labels: t_labels}
                _,l = session.run([optimizer,loss], feed_dict = feed_dict)
                total_loss+=l
                start_index +=1

                if step%10000==0 and step!=0:
                    #Create checkpoint
                    if os.path.isfile("wv2.ckpt"):
                        os.remove("wv2.ckpt")
                        checkpoint = saver.save(session, "wv2.ckpt")
                if step%epoch_size==0 and step!=0:
                    loss_values.append(total_loss/epoch_size)
                    total_loss=0


            embeddings = normalized_embeddings.eval()
            print("Training done!")
        
        #Print loss plot
        pylab.ylabel("Loss")
        pylab.xlabel("Step #")
        loss_value_array = np.array(loss_values)
        print("SHAPES! loss, other")
        print(loss_value_array.shape)
        pylab.plot(np.arange(1,100000, 1001),loss_values)
           
        pylab.show()            

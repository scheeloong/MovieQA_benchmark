
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

import tensorflow as tf
import numpy as np

from matplotlib import pylab
from six.moves import range
from six.moves.urllib.request import urlretrieve
#from sklearn.manifold import TSNE


from collections import OrderedDict, defaultdict, Counter, deque


vocabulary = Counter() #Map of word to # instances 

path = '/nfs/ug/homes-3/v/vukovic8/Desktop/ECE496/MovieQA_benchmark-master/story/plot'
#Index in 1-hot vector
hot_indices_map = {'UNK':0}
hot_indices = []

#Contains all data from all movies
all_text_data = []



#MovieQA files
def load_files():
    """Opens all files and loads into dict"""
    """Also loads all files into single string for training"""
    parsed_movie_dict = {}
    #Get the files
    file_list= os.listdir(path)


    for file_i in file_list:
        filename = file_i
        with open(path + '/' + filename) as file:
            text = file.read()

            parsed_file = word_parse(text)

            parsed_movie_dict[filename] = parsed_file

            #Add to list of all movie text
            all_text_data.extend(parsed_file)

 
    
    return (parsed_movie_dict, all_text_data)  


def word_parse(input_text):
    """given a file, remove all punctuation/capitalization and 
    place all words into a list.
    """
    word_list = []

    #Remove all capitalization from input text
    input_text = input_text.lower()
    input_text = input_text.replace(',', '')
    input_text = input_text.replace('\n', ' ')
    input_text = input_text.replace(':', '')
    input_text = input_text.replace(';', '')
    input_text = input_text.replace('(', '')
    input_text = input_text.replace(')', '')
    input_text = input_text.replace('!', '')
    input_text = input_text.replace('?', '')
    input_text = input_text.replace('.', '')


    #Replace multiple spaces?

    word_list = input_text.split()

    
    return word_list

def generate_vocabulary(parsed_movie_dict):
    '''Add words to vocabulary and update word frequency'''
    for movie in parsed_movie_dict:

        for word in parsed_movie_dict[movie]:

            if word in vocabulary:
                vocabulary[word] +=1
            else:
                vocabulary[word] = 1



def vocab_to_one_hot(recognized_words):
    
    for _, word in enumerate(vocabulary):

        if word in recognized_words:
            index = len(hot_indices)
            hot_indices_map[word] = index
            hot_indices.append(index)
        else:
            #Unknown word
            hot_indices_map[word] = 0


#Vocab setup
(parsed_movie_dict, all_text_data) = load_files()
generate_vocabulary(parsed_movie_dict)

VOCABULARY_SIZE = 10000
MIN_FREQ = 5 #Minimum number of times a words must appear to be recognized
frequent_words = vocabulary.most_common(VOCABULARY_SIZE)
recognized_words = []
#Take VOCABULARY_SIZE most common words
for word, _ in frequent_words:
    recognized_words.append(word)

#Create 1-hot mapping
vocab_to_one_hot(recognized_words)

#Get one-hot-indexed representation of all the text 
data = [] #Change to array?
for word in all_text_data:
    if word in hot_indices_map:
        #Recognized
        data.append(hot_indices_map[word])
    else:
        #Not recognized
        data.append(hot_indices_map['UNK'])
    


#NOTE TO SELF: FIX GENERATE_BATCH -----------------
def generate_batch(num_skips, batch_size, data, start_index):
    '''Generates a training batch from the given file'''
    #Note assumes that num_skips will also be the max distance from the current word
    #Data is the one-hot index representation of the words in the vocabulary
    #num_skips is number of skips to do from each center word
    #start index acts as offset
    
    assert batch_size%num_skips == 0 #Ensure batch size is divisible by number of skips
    assert num_skips%2 ==0
    window_size = num_skips +1

    data_len = len(data)
    num_center_words = batch_size/num_skips #number of words to skip from

    #Pick out subset of data to be used for this batch
    num_data_words = num_center_words + num_center_words + num_skips

    if (start_index + num_data_words)>= (data_len-1):
        #Insufficient words left in text data, loop back to beginning
        num_extra_words = (start_index + num_data_words) - (data_len-1)
        curr_data.append(data[start_index:data_len])
        curr_data.append(data[0:num_extra_words])
    else:
        curr_data = data[start_index: num_data_words]

    batch = np.ndarray(shape=(batch_size), dtype=np.int32) #Holds the center words of current training batch
    labels = np.ndarray(shape=(batch_size), dtype=np.int32) #holds the labels of the current training batch
    
    for i in range(num_center_words):
        #Fill training batch/labels
        center_word = num_skips/2 + i
        for j in range(num_skips/2):
            batch[num_skips*i +2*j] = curr_data[center_word]
            batch[num_skips*i +2*j+1] = curr_data[center_word]

            #Words j to the left and right of center_word
            labels[num_skips*i +2*j] = curr_data[center_word - j-1]
            labels[num_skips*i +2*j+1] = curr_data[center_word +j +1]
    
           

    return(batch,labels)


#Graph parameters
window_size = 2
num_skips = 4
batch_size = 128
embedding_dim = 128 #Embedding vector dimension
start_index = 0
num_sampled = 64 #Netagive examples

graph = tf.Graph()



#NOTE TO SELF: FIX GENERATE BATCH FIRST!!!! (test.py)
with graph.as_default(),tf.device('/cpu:0'):
    #Initial loss
    loss = 0
    

    #The training data (center words) and training labels (predictions)
    train_data = tf.placeholder(tf.int32, shape=[batch_size])
    train_label = tf.placeholder(tf.int32, shape=[batch_size,1]) #Check shape (num_skips vs 1)
    
    embeddings = tf.Variable(tf.random_uniform([VOCABULARY_SIZE, embedding_dim], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, train_data)
    weights = tf.Variable(tf.truncated_normal([VOCABULARY_SIZE, embedding_dim], stddev=1.0 / math.sqrt(embedding_dim)))

    #DOUBLE CHECK BIASES
    biases = tf.Variable(tf.zeros([VOCABULARY_SIZE]))

    
    #Save model variables
    loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights, biases, embed, train_label, num_sampled, VOCABULARY_SIZE))
    #save = tf.train.Saver()

    #Gradient Descent
    optimizer = tf.train.AdagradOptimizer(0.001).minimize(loss)

    #Cosine Similarity
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings),1, keep_dims=True))
    normalized_embeddings = embeddings/norm

    #Validation embeddings?

    


#For plotting the loss
loss_values = []

#TRAINING GOES HERE
with tf.Session(graph=graph) as session:
    #Initialize variables
    tf.initialize_all_variables().run()
    print('Variables initialized')

    num_steps = 100001
    start_index = 0
    total_loss = 0

   

    for step in range(num_steps):
        batch_data, batch_labels = generate_batch(num_skips, batch_size, data, start_index)
        #print(batch_labels)
        batch_labels = np.reshape(batch_labels,(128,1))
        #print(type(batch_labels))

        feed_dict = {train_data: batch_data, train_label: batch_labels}
        #print(feed_dict[train_label])
        _,l = session.run([optimizer,loss], feed_dict = feed_dict)

        total_loss+=l

        #Estimate loss
        if step %1000 ==0 and step !=0:
            
            loss_values.append(total_loss/1000)
           
            total_loss = 0

        #Save data?
        start_index =+1

    embeddings = normalized_embeddings.eval()
    print("Completed")

    pylab.ylabel("Loss")
    pylab.xlabel("Step #")
    pylab.plot(np.arange(1,100000, 1001),loss_values)
           
    pylab.show()            




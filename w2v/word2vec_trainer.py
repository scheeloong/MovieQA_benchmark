import os
import math
import numpy as np
import tensorflow as tf
import string
import random
import zipfile
import pickle

from six.moves.urllib.request import urlretrieve
from collections import Counter, deque
from w2v_utils import tokenize_text, load_text8, load_plots, save_obj, get_batch

get_ipython().magic('matplotlib inline')

class Word2Vec():
    def __init__(self, name="testing", size=40000, thresh=3):
        self.name = name
        ckpt_file = "w2v_%s.ckpt" % self.name
        id_map_file = "id_map_%s.pkl" % self.name
        embed_file = 
        exclude = '!"#$%&()*+,./:;<=>?@[\]^_`{|}~-' + "'"

        self.vocab = Counter()
        self.id_map = {'UNK': 0}

        text8, t8_count = load_text8()
        plots, plot_count = load_plots()
        self.vocab = t8_count + plot_count
        print(text8[:10])

        # Add all plot words above threshold to 
        # self.vocab id map.
        '''
        for plot in plots:
            for word in plot:
                if word not in self.id_map and self.vocab[word] > thresh:
                    self.id_map[word] = len(self.id_map)
        '''
        common_words = self.vocab.most_common(size)
        # Add common words in text8 to self.vocab, up to the self.vocab size
        for word, _ in common_words:
            '''if len(self.id_map) >= VOCAB_SIZE:
                break'''
            if word not in self.id_map:
                self.id_map[word] = len(self.id_map)


        # Tokenize texts
        tokenized_text8 = tokenize_text(text8, self.id_map)

        
        # Change this to the data you want to use!
        self.data = np.array(tokenized_text8) # or use tk_plot
        #data = np.array(plt_train)
        #data = np.array(tokenized_text8 + tk_plot)

        #tk_plot = []
        #for plot in plots:
        #    tk_plot.extend(tokenize_text(plot, self.id_map))

        # Split for train val test
        #t8_len = len(tokenized_text8)
        #t8_train = tokenized_text8[:math.floor(t8_len*.7)]
        #t8_val = tokenized_text8[math.floor(t8_len*.7) :math.floor(t8_len*.8)]
        #t8_test = tokenized_text8[math.floor(t8_len*.8):]

        # Split for train val test
        #plt_len = len(tk_plot)
        #plt_train = tk_plot[:math.floor(plt_len*.7)]
        #plt_val = tk_plot[math.floor(plt_len*.7) :math.floor(plt_len*.8)]
        #plt_test = tk_plot[math.floor(plt_len*.8):]

        save_obj(self.id_map, id_map_file)

        VOCAB_SIZE = len(self.id_map)
        print('Vocab Size:', len(self.id_map))
        print('Common words cutoff', common_words[-1])


    def build_graph(self):
        dim = 128 # number of dimensions of representation
        vocab_size = len(self.id_map)
        num_sampled = 15
        batch_size = 128
        window_size = 10

        graph = tf.Graph()

        with graph.as_default():
            word_ids = tf.placeholder(tf.int64, shape=[batch_size])
            labels = tf.placeholder(tf.int64, shape=[batch_size, window_size*2])
            embed = tf.Variable(tf.random_uniform([vocab_size, dim], -1, 1), name='center_rep')
            h1 = tf.nn.embedding_lookup(embed, word_ids)
            
            loss = 0
            
            with tf.device('/cpu:0'):
                for i in range(window_size*2):
                    w = tf.Variable(tf.truncated_normal([vocab_size, dim], stddev=1.0 / math.sqrt(dim)), name='context_rep')
                    b = tf.Variable(tf.zeros([vocab_size]))
                    loss += tf.reduce_mean(tf.nn.sampled_softmax_loss(w, b, h1,  labels=labels[:,i:i+1], 
                                               num_sampled=num_sampled, num_classes=vocab_size))
                
            optimizer = tf.train.AdagradOptimizer(1).minimize(loss)
            saver = tf.train.Saver()
            
            norm = tf.sqrt(tf.reduce_sum(tf.square(embed), 1, keep_dims=True))
            norm_embeddings = embed/norm

        return graph, optimizer, loss, norm_embeddings


    def train(self, graph, optimizer, loss, embedding_tensor, num_steps=100001):
        # Run Training
        num_steps = 100001
        save_path = ""

        # Session config
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.5
        config.gpu_options.allow_growth = True


        loss_data = []
        with tf.Session(graph=graph, config=config) as sess:
            # Initialize
            tf.initialize_all_variables().run()
            print("vars init.")
            if os.path.isfile(ckpt_file):
                print("Loading from ckpt file")
                saver.restore(sess, ckpt_file)     
            
            # Run
            sum_l = 0
            print("Training")
            for step in range(num_steps):
                batch_in, batch_lbls = get_batch(step, self.data, batch_size, window_size)
                    
                feed_dict = {word_ids: batch_in, labels: batch_lbls}
                _, l = sess.run([optimizer, loss], feed_dict=feed_dict)
                sum_l += l
                
                if step%4000 == 0 and step!=0:
                    loss_data.append(sum_l/4000)
                    print(str(step) + ": "+ str(sum_l/4000))
                    sum_l=0
                    if step%52000 == 0:
                        save_path = saver.save(sess, ckpt_file)
                    

            if os.path.isfile(ckpt_file):
                os.remove(ckpt_file)
            save_path = saver.save(sess, ckpt_file)
            embed = embedding_tensor.eval()
            print("Finished Training")


        # Save word embeddings. 
        embed_file = "embed_%s" % self.name
        np.savez(embed_file, embed = embed)

        print("Size of embedding file:", os.path.getsize(save_path))


    def tsne_plot():

        from matplotlib import pylab
        from sklearn.manifold import TSNE
        num_points = 1000
        common_words = [x[0] for x in self.vocab.most_common(num_points)]
        common_indexes = []
        for i in range(num_points):
            common_indexes.append(self.id_map[common_words[i]])
        common_indexes = np.array(common_indexes)

        embed[embed<0.001] = 0.001
        print embed[:20,:]


        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        #two_d_embeddings = tsne.fit_transform(embed[400:400+num_points+1, :])
        two_d_embeddings = tsne.fit_transform(embed[common_indexes, :])

        def plot(embeddings, labels):
          assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'
          pylab.figure(figsize=(15,15))  # in inches
          for i, label in enumerate(labels):
            x, y = embeddings[i,:]
            pylab.scatter(x, y)
            pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
                           ha='right', va='bottom')
          pylab.show()

        reverse_dictionary = dict(zip(self.id_map.values(), self.id_map.keys())) 
        words = [reverse_dictionary[i] for i in range(400, 400+num_points+1)]
        plot(two_d_embeddings, words)


    # In[28]:
    def loss_plot():
        pylab.ylabel("loss")
        pylab.xlabel("steps")
        pylab.plot(np.arange(1,num_steps-1, 4001), loss_data)

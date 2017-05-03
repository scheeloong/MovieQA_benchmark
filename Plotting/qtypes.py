import os
import math
import random
import string
import zipfile
import pdb

import random

import numpy as np
import sys
from matplotlib import pylab as plt
from matplotlib import cm
from collections import Counter
from collections import OrderedDict


if __name__=="__main__":
    #Open text file containing all annotated question and anwers
    annotated_q_file = open('./movieqa_answer_types/ans_type.txt','r')

    curr_line = annotated_q_file.readline()
    qu_type_dict = {'a':0}
    question_counter = Counter(qu_type_dict)
    
    while curr_line != '':
        #Get list of current question types
        curr_line = curr_line.split(':')[1]
        curr_line = curr_line.replace(' ', '')
        curr_line = curr_line.replace('\n', '')
        
        curr_line_qu = ''.join(c for c in curr_line if not c.isdigit())
        curr_count = Counter(curr_line_qu)
        question_counter.update(curr_count)
        curr_line = annotated_q_file.readline()


        

    #List of question types
    q_a = 'a = Person/Org (who)\n'
    q_b = 'b = PType/Relation (who)'
    q_c = 'c = Object or Thing (what)'
    q_d = 'd = Action (what)'
    q_e = 'e = Abstract (what)'
    q_f = 'f = Objective (what)'
    q_g = 'g = Location (where)'
    q_h = 'h = Event/Time (when)'
    q_i = 'i = Yes/No (is,does)'
    q_j = 'j = Choice (which)'
    q_k = 'k = Causality (happen)'
    q_l = 'l = R rated'
    q_n = 'm = Reason-action (how)'
    q_o = 'n = Count (how many)'
    q_p = 'o = Age/Time (old)'
    q_q = 'p = Length (long/far)'
    q_r = 'q = Emotion (how feel)'
    q_s = 'r = Other'
    q_t = 's = Dialog'


    sorted_qu_dict = OrderedDict(sorted(question_counter.items()))
    #Plots
    labels = sorted_qu_dict.keys()
    sizes = sorted_qu_dict.values()


    #Set colours
    num_slices = len(labels)
    col = np.random.random(num_slices)
    colors =cm.Set1(np.arange(num_slices)/float(num_slices))

    #Set spacing
    explode = [0.15]*num_slices

    plt.pie(sizes, colors = colors, labels=labels, explode=explode, autopct='%1.1f%%', shadow=True, startangle=0)

    #plt.legend(Legend_string, labels2, loc = "best") 
    plt.axis('equal')
    plt.show()

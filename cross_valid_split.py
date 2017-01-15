import math
import numpy as np
import random

def cross_valid_split(dataset, total_num_folds, curr_fold_num, split_ratios, clip_size=0):
    '''Split the dataset into train, test, val sets. 
       split_ratios is a list of [train_set_percentage, test_set_percentage, valid_set_percentage].
       clip_size is an optional parameter determining how much of the dataset ends to clip off.'''

    [train_set_split, test_set_split, valid_set_split] = split_ratios
    
    if clip_size!=0:
        usable_dataset = dataset[clip_size:-clip_size]
    else:
        usable_dataset=dataset
    dataset_size = len(usable_dataset)
    
    assert (total_num_folds<dataset_size) #To avoid training on the same split more than once 
    fold_offset = int(math.floor(dataset_size/total_num_folds)) #how much each fold is offset from previous iteration 

    train_set_size = int(math.floor(dataset_size*train_set_split))
    test_set_size = int(math.floor(dataset_size*test_set_split))
    valid_set_size = int(math.floor(dataset_size*valid_set_split))

    train_set_start_index = fold_offset*curr_fold_num
    test_set_start_index = (train_set_start_index + train_set_size)%dataset_size
    valid_set_start_index = (test_set_start_index + test_set_size)%dataset_size
    #valid_set_stop_index = (valid_set_start_index + valid_set_size +1)%dataset_size

    #TODO: consider using np arrays instead
    if test_set_start_index<train_set_start_index:
        #train set loops around
        train_set = usable_dataset[train_set_start_index:] + usable_dataset[:test_set_start_index]
        test_set = usable_dataset[test_set_start_index:valid_set_start_index]
        valid_set = usable_dataset[valid_set_start_index:train_set_start_index]
        print('loop1')
    elif valid_set_start_index<test_set_start_index:
        #test set loops around
        train_set = usable_dataset[train_set_start_index:test_set_start_index]
        test_set = usable_dataset[test_set_start_index:] + usable_dataset[:valid_set_start_index]
        valid_set = usable_dataset[valid_set_start_index:train_set_start_index]
        print('loop2')
    elif train_set_start_index<valid_set_start_index:
        #validation set loops around
        train_set = usable_dataset[train_set_start_index:test_set_start_index]
        test_set = usable_dataset[test_set_start_index:valid_set_start_index]
        valid_set = usable_dataset[valid_set_start_index:] + usable_dataset[:train_set_start_index]
        print('loop3')
    else:
        #No looping
        train_set = usable_dataset[train_set_start_index:test_set_start_index]
        test_set = usable_dataset[test_set_start_index:valid_set_start_index]
        valid_set = usable_dataset[valid_set_start_index:]
        print('loop4')   
    #random.shuffle(train_set)

    return (train_set, test_set, valid_set)


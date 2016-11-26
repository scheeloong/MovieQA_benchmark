#Backup of batch generation by file
def generate_batch(num_skips, batch_size, data):
    '''Generates a training batch from the given file'''
    #Note assumes that num_skips will also be the max distance from the current word
    #Data is the one-hot index representation of the words in the vocabulary
    #num_skips is number of skips to do from each center word
    
    assert batch_size%num_skips == 0 #Ensure batch size is divisible by number of skips
    assert num_skips%2 ==0

    data_len = len(data)
    num_center_words = batch_size/num_skips #number of words to skip from
    print(data_len)
    print (num_center_words)
    print (data_len - num_skips/2 + 2)
    

    batch = np.ndarray(shape=(batch_size), dtype=np.int32) #Holds the center words of current training batch
    labels = np.ndarray(shape=(batch_size), dtype=np.int32) #holds the labels of the current training batch

    for i in range(num_center_words):
        #Fill training batch/labels
        center_word = num_skips/2 + i
        for j in range(num_skips/2):
            batch[num_skips*i +2*j] = data[center_word]
            batch[num_skips*i +2*j+1] = data[center_word]
            
            labels[num_skips*i +2*j] = data[center_word - j-1]
            labels[num_skips*i +2*j+1] = data[center_word +j +1]
    
           

    return(batch,labels)

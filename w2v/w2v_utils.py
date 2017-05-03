import os
import pickle
from collections import Counter

exclude = '!"#$%&()*+,./:;<=>?@[\]^_`{|}~-' + "'"

def tokenize_text(words, id_map):
    ''' Convert list of words into list of tokens/word-ids.
    Args:
      id_map: Hashmap from word to id. id of 0 is always 'UNK' unkown token.
    Ret: list of ints (ids)
    '''
    tokens = []
    for i, word in enumerate(words):
        id = 0  # 0 is UNK id
        if word in id_map:
            id = id_map[word]
        tokens.append(id)

    return tokens


def count_words(func):
    def load_file(*args, **kwargs):
        text_data = func(*args, **kwargs)
        word_count = Counter()
        for item in text_data:
            word_count[item] += 1
        return text_data, word_count

    return load_file


@count_words
def load_plot(file_name):
    with open(file_name) as f:
        t = f.read().lower()
        # Filter punctuation
        plot_words = ''.join(
            ch if ch not in exclude else ' ' for ch in t).split()
    return plot_words


@count_words
def load_text8(file_name="text8"):
    ''' Load already cleaned data file
    '''
    text8 = []
    with open(file_name) as f:
        text8 = f.read().split()

    return text8


def load_plots(plot_dir="plot/"):
    ''' Load raw plot data file.
    '''
    plot_files = os.listdir(plot_dir)
    plots = []
    word_count = Counter()

    for plot_file in plot_files:
        plot_words, count = load_plot(plot_dir + plot_file)
        word_count += count
        plots.append(plot_words)

    return plots, word_count


def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    print('%s saved' % name)


def get_pairs(start_index, data, batch_size=128, wing_size=10):
    ''' start: 
        data:
        size: batch size
        
        returns numpy array of input data and numpy array of labels
    '''
    index = start_index%len(data)
    batch_in = []
    batch_lbls = []
    window_size = wing_size*2+1
    
    context = [data[(index+offset)%len(data)] for offset in range(-1*wing_size, wing_size+1)]
    context = deque(context, maxlen=window_size)
    for i in range(batch_size):
        # Loop through words
        batch_in.append(data[index])
        batch_lbls.append(context)

        index = (index+1)%len(data)
        
        context.append(data[(index+wing_size)%len(data)])
        
    return np.array(batch_in), np.array(batch_lbls)[:,np.arange(window_size)!=wing_size]


def get_batch(batch_num, data, batch_size, wing_size=10):
    init = (batch_num*batch_size)%len(data)
    return getPairs(init, data, batch_size, wing_size)

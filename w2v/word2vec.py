import numpy as np
import pickle
import MovieQA


class Word2Vec(object):
    def __init__(self, extension='plt', postprocess=False):
        # Filenames & Constants
        ID_MAP_FILE = "id_map_%s.pkl" % extension
        EMBED_FILE = "embed_%s.npz" % extension
        self.SYMBOLS_TO_REMOVE = '"#$%&()*+,/:;<=>@[\]^_`{|}~-' + "'?!"

        # Get word2vec embeddings
        self.embeddings = []
        with np.load(EMBED_FILE) as f:
            self.embeddings = f['embed']
            print('Embeddings loaded')

        # Get the word to word-id mappings
        self.id_map = self.load_obj(ID_MAP_FILE)

        if postprocess:
            self._postprocess()

    def load_obj(self, name):
        with open(name, 'rb') as f:
            return pickle.load(f)

    def normalize(self, mat):
        if mat.ndim == 1:
            return mat / (np.linalg.norm(mat) + 1e-6)
        return mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-6)

    def tokenize_text(self, words):
        ''' Convert cleaned text into list of tokens/word-ids.
        Ret: list of ints (ids)
        '''
        words = words.split()
        tokens = []
        for i, word in enumerate(words):
            id = 0  # 0 is UNK id
            word = word.lower()
            if word in self.id_map:
                id = self.id_map[word]
            tokens.append(id)

        return np.array(tokens)

    def _clean_text(self, text_raw):
        ''' Splits sentences and then filters.
        '''
        text = text_raw
        if isinstance(text_raw, list):
            text = ' '.join(sentence for sentence in text_raw)
        text = filter(lambda x: x not in self.SYMBOLS_TO_REMOVE, text)
        text = text.split(". ")
        return text

    def _filter_sentence(self, sentence_raw):
        ''' Filters out unused symbols including period.
        '''
        text = filter(lambda x: x not in (
            self.SYMBOLS_TO_REMOVE + '.'), sentence_raw)
        return text

    def _postprocess(self):
        ''' From inarXiv:1702.01417v1 Paper
        '''
        from sklearn.decomposition import PCA
        # Remove non-zero mean.
        vocab_size, n_dim = self.embeddings.shape
        mu = np.sum(self.embeddings, axis=0, keepdims=True) / vocab_size
        self.embeddings = self.embeddings - mu

        # PCA
        D = int(n_dim / 100)
        pca = PCA(n_components=D)
        pca.fit(self.embeddings)

        # Find dominating directions.
        u = np.matmul(self.embeddings, pca.components_.T)
        dom_directions = np.matmul(u, pca.components_)
        self.embeddings = self.embeddings - dom_directions

        print "Finished postprocessing."

    def get_sentence_vector(self, sentence):
        clean_sentence = self._filter_sentence(sentence)
        sentence_vector = self.embeddings[self.tokenize_text(clean_sentence)]
        normalized_sentence_vector = self.normalize(
            np.average(sentence_vector, axis=0))
        # print(normalized_sentence_vector.shape)
        return normalized_sentence_vector

    def get_vectors_for_raw_text(self, text):
        ''' Get a matrix of embeddings for a text with multiple sentences (i.e. plot).
        '''
        cleaned_text = self._clean_text(text)
        return self.get_text_vectors(cleaned_text)

    def get_text_vectors(self, cleaned_text):
        ''' Get a matrix of embeddings for a text with multiple sentences (i.e. plot).
            Assume text has already been cleaned.
        '''
        embedding_matrix = np.array([self.get_sentence_vector(line)
                                     if line != ""
                                     else np.zeros(self.embeddings.shape[1])
                                     for line in cleaned_text])

        return embedding_matrix

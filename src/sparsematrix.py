from scipy.sparse import dok_matrix

import numpy as np

class SparseMatrix(object):

    def __init__(self, vocabulary, movieKeys):
        '''
        Initialize the sparse matrix with the given vocabularyh and keys
        '''

        # A sparse matrix representing (vocabulary x movies)
        self.sparseMatrix  = dok_matrix((len(vocabulary), len(movieKeys)), dtype=np.float32)

        # A map from unique word to index
        self.wordToIndex = {}

        # A map from unique movie to index
        self.movieToIndex = {}

        # Build Word Index
        self.uniqueWordIndex = 0
        for uniqueWord in vocabulary:
            self.wordToIndex[uniqueWord] = self.uniqueWordIndex
            self.uniqueWordIndex += 1

        # Build Movie Index
        self.uniqueMovieIndex = 0
        for uniqueMovieKey in movieKeys:
            self.movieToIndex[uniqueMovieKey] = self.uniqueMovieIndex
            self.uniqueMovieIndex += 1

    def getMatrix(self):
        return self.sparseMatrix

    def getScore(self, word, movie):
        return self.sparseMatrix[self.wordToIndex[word], self.movieToIndex[movie]]

    def setScore(self, word, movie, score):
        self.sparseMatrix[self.wordToIndex[word], self.movieToIndex[movie]] = score

    def getWordIndex(self, word):
        return self.wordToIndex[word]

    def contains(self, word, movie):
        '''
        Returns True if the sparseMatrix contains the given (word, movie) score
        and false otherwise
        '''
        if word in self.wordToIndex and movie in self.movieToIndex:
            return self.sparseMatrix.has_key((self.wordToIndex[word], self.movieToIndex[movie]))
        return False

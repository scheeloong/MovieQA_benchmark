from scipy.sparse import dok_matrix # TODO: Build with dsr matrix then convert to csr or csc.

import numpy as np

# TODO: Maybe use defaultdict from collections if it's useful
class SparseMatrix(object):
    def __init__(self, vocabulary, movieKeys):
        # A map from unique word to index
        self.wordToIndex = {}
        # A map from unique movie to index
        self.movieToIndex = {}

        # A sparse matrix representing (vocabulary x movies)
        self.sparseMatrix  = dok_matrix((len(vocabulary), len(movieKeys)), dtype=np.float32)

        # Build Word Index
        self.uniqueWordIndex = 0
        for word in vocabulary:
            self.wordToIndex[word] = self.uniqueWordIndex
            self.uniqueWordIndex += 1

        # Build Movie Index
        self.uniqueMovieIndex = 0
        for movieKey in movieKeys:
            self.movieToIndex[movieKey] = self.uniqueMovieIndex
            self.uniqueMovieIndex += 1

    def getWordIndex(self, word):
        return self.wordToIndex[word]

    def getScore(self, word, movie):
        return self.sparseMatrix[self.wordToIndex[word], self.movieToIndex[movie]]

    def setScore(self, word, movie, score):
        self.sparseMatrix[self.wordToIndex[word], self.movieToIndex[movie]] = score

    def getMatrix(self):
        return self.sparseMatrix

    def contains(self, word, movie):
        if word in self.wordToIndex and movie in self.movieToIndex:
            return self.sparseMatrix.has_key((self.wordToIndex[word], self.movieToIndex[movie]))
        return False

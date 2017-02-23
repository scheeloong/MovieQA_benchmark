import math
import numpy as np
import logging
from src.movietokenizer import MovieTokenizer
from src.sparsematrix import SparseMatrix

class TfIdf(object):
    """
    This class implements the
    term frequency-inverse document frequency
    algorithm that is used for word embedding

    Analysis:
        Time Complexity, T(V,M) = O(VM)
          Iterate through every movie and every word once since we must fill up
          the 2D matrix of size (vocabularySize * numberOfMovies)
          Therefore, this will be the fastest approach to train the TFIDF word embedding matrix.
          It also does not normalize the TFIDF score as they are rank invariant.
        Space Complexity, S(V,M) = O(V)
          Idfvec = 12k * 1 = 12k memory
          Store counter for each movie for each iteration => 12k * 1
          and override since won't need to read it again after each movie.
    """

    def __init__(self, story):
        """
        Story are the list of all available plots for the movie.
        Each plot contains a number of
        """

        # Tokenize alphanumeric with dash, lowercase, stem.
        self.tokenizer = MovieTokenizer("[\w]+")

        # All the stories
        # [keyForStory, allPlotsForStory]
        self.story = story

        # A set for all possible words in vocabulary
        self.vocabularySet = set()

        # A map for each movie to its vocabulary set
        self.movieToVocabularySet = {}

        self.initVocabulary()

        self.numberOfStories = len(self.story)
        self.numberOfWords = len(self.vocabularySet)

        # A sparse matrix for tfIdfMatrix
        self.tfIdfMatrix = SparseMatrix(self.vocabularySet, self.story)
        self.idfVec = np.zeros(self.numberOfWords)
        for currWord in self.vocabularySet:
            self.idfVec[self.tfIdfMatrix.getWordIndex(currWord)] = self.inverseDocumentFrequency(currWord)

        # Remove movieToVocabularySet as it is no longer needed
        # It was only used for IDF calculation
        self.movieToVocabularySet = {}

        self.initTfIdfMatrix(0.0)
        # The matrix dimension is (|Vocabulary Size| * numberOfStories)
        # But excluded those with tfidf score below a certain threshold
        """
                                movieId_1       ...    movieId_numberOfMovies
                                ____________|_______|________________________
        word_1             | tfidf(word_1, movieId_1)            |       |
        ...                |                |       |
        word_numberOfWords |                |       |
        """

        logging.info('Number of Stories: ' + str(self.numberOfStories))
        logging.info('Number of Words: ' + str(self.numberOfWords))

    def initTfIdfMatrix(self, tfIdfThreshold):
        """ This method trains an entire new tfIdfMatrix. """
        for currMovieKey in self.story:
            countOfWords = self.tokenizer.tokenizeDuplicatePerSentence(self.story[currMovieKey])
            for word in countOfWords:
                tfScore = np.log10(countOfWords[word]) + 1.0
                idfScore = self.idfVec[self.tfIdfMatrix.getWordIndex(word)]
                tfIdfScore = tfScore * idfScore
                if tfIdfScore >= tfIdfThreshold:
                    self.tfIdfMatrix.setScore(word, currMovieKey, tfIdfScore)

    def initVocabulary(self):
        """
        Initializes both self.vocabularySet and self.movieVocabularySet
        """
        for currMovieKey in self.story:
            currMoviePlots = self.story[currMovieKey]
            self.movieToVocabularySet[currMovieKey] = set()
            for currPlot in currMoviePlots:
                self.movieToVocabularySet[currMovieKey].update(self.tokenizer.tokenizeAlphanumericLower(currPlot))
            self.vocabularySet.update(self.movieToVocabularySet[currMovieKey])

    def inverseDocumentFrequency(self, word):
        """
        Returns the inverse document frequency of the
        given word that is calculated from all stories.
        """
        count = 0.0
        for currMovieKey in self.story:
            if word in self.movieToVocabularySet[currMovieKey]:
                count += 1.0
        # If it doesn't exist from the vocabulary, return 0.0
        if count == 0.0:
            return count
        return np.log10(len(self.story)/count)

    def getSentenceVector(self, movieKey, sentence):
        sentenceVec = np.zeros(self.numberOfWords)
        cleanSentence = self.tokenizer.tokenizeAlphanumericLower(sentence)
        for currWord in cleanSentence:
            if self.tfIdfMatrix.contains(currWord, movieKey):
                sentenceVec[self.tfIdfMatrix.getWordIndex(currWord)] = self.tfIdfMatrix.getScore(currWord, movieKey)
        return sentenceVec

    def getSentenceVectors(self, movieKey, sentences):
        ''' Sentences is a list of strings.
        '''
        embedding_matrix = np.zeros((len(sentences),self.numberOfWords))
        for i, sentence in enumerate(sentences):
            embedding_matrix[i] = self.getSentenceVector(movieKey, sentence)
        return embedding_matrix

    def getWordScoreArray(self, movieKey, sentence):
        ''' sentence is a string.
        '''
        cleanSentence = self.tokenizer.tokenizeOrderedAlphanumericLower(sentence)
        sentenceVec = np.zeros((len(cleanSentence),1))
        for i, currWord in enumerate(cleanSentence):
            if self.tfIdfMatrix.contains(currWord, movieKey):
                sentenceVec[i] = self.tfIdfMatrix.getScore(currWord, movieKey)

        return sentenceVec


import math
import numpy as np
import tensorflow as tf
import pickle

# TODO: Replace all instances of shouldLog with a proper logger
shouldLog = True

class TfIdf(object):
    """
    This class implements the
    term frequency-inverse document frequency
    algorithm that is used for word embedding
    It serializes the trained words using pickle if it isn't already trained
    and deserialize the trained words from pickle"""

    # TODO: Implement serializing, checking if serialized to skip serialization
    # TODO: Implement deserializing
    def __init__(self, story, loadFile=None):
        """
        Story are the list of all available plots for the movie.
        Each plot contains a number of
        """
        # All the stories 
        #   Format is:
        #       [keyForStory, allPlotsForStory]
        self.story = story

        # The number of stories in this class
        self.numberOfStories = len(self.story)

        # A set for all possible vocabulary
        self.vocabularySet = self.getVocabulary()

        # A map for each movie to it's vocabulary set
        # Used by IDF score calculation
        self.movieToVocabularySet = {}

        for currMovieKey in self.story:
            self.movieToVocabularySet[currMovieKey] = self.getVocabularyForMovie(currMovieKey)

        # Number of distinct words in the vocabularySet
        self.numberOfWords = len(self.vocabularySet)

        print 'Number of stories: ' + str(self.numberOfStories)
        print 'Number of Words: ' + str(self.numberOfWords)
        # arr[words][movieId]

        # Get the tfIdfMatrix
        if loadFile is None:
            self.tfIdfMatrix = self.getTfIdfMatrix(0.001)
        else:
            self.tfIdfMatrix = self.loadTfIdfMatrix(loadFile)
        if shouldLog:
            print self.tfIdfMatrix
        # The matrix dimension is (|Vocabulary Size| * numberOfStories)
        # But excluded those with tfidf score below a certain threshold
        """
                                movieId_1       ...    movieId_numberOfStories
                                ____________|_______|________________________
        word_1             | tfidf(word_1, movieId_1)            |       |  
        ...                |                |       |
        word_numberOfWords |                |       |
        """


    def getTfIdfMatrix(self, tfIdfThreshold):
        fileName = self.saveTfIdfMatrix(tfIdfThreshold)
        return self.loadTfIdfMatrix(fileName)

    def loadTfIdfMatrix(self, fileName):
        """ 
        This method loads the already processed tfIdfMatrix using pickle.
        """
        with open(fileName, 'rb') as fp:
            return pickle.load(fp)

    def saveTfIdfMatrix(self, tfIdfThreshold):
        """
        This method trains an entire new tfIdfMatrix and serializes it to file.
        """
        tfIdfMatrix = {}
        # TODO: Clean up words with nltk (get rid of ',' and match similar words)
        count = 0
        limitVocabForTesting = 15000 # TODO: Set this as a parameter
        for currWord in self.vocabularySet:
            if shouldLog:
                print "Training word: " + currWord
            exist = False
            if count > limitVocabForTesting:
                break
            idfScore = self.inverseDocumentFrequency(currWord)
            # Skip this word if it is useless ('the')
            if (idfScore < tfIdfThreshold):
                continue
            # Calculate the tfidf score for every word
            for currMovieKey in self.story:
                # Calculate the tfScore for this word in this movie
                tfScore = self.termFrequency(currWord, self.story[currMovieKey])
                if tfScore == 0.0:
                    continue
                tfIdfScore = tfScore * idfScore
                if shouldLog:
                    print "tfscore: " + str(tfScore)
                    print "idfscore: " + str(idfScore)
                    print "tfidfscore: " + str(tfIdfScore)
                if tfIdfScore >= tfIdfThreshold:
                    tfIdfMatrix[currWord,currMovieKey] = tfIdfScore
                    exist = True
            if exist:
                count += 1.0
        with  open('tfIdfMatrix.obj', 'wb') as fp:
            pickle.dump(tfIdfMatrix, fp)
        return 'tfIdfMatrix.obj'

    def getVocabularyForMovie(self, currMovieKey):
        """
        Returns a set of all vocabulary for a given movie
        """
        vocabulary = set()
        currMoviePlots = self.story[currMovieKey]
        for currPlot in currMoviePlots:
            for word in currPlot.split():
                vocabulary.add(word)
        return vocabulary

    def getVocabulary(self):
        """
        Returns all the distinct words from all movies as a set
        """
        vocabulary = set()
        for currMovieKey in self.story:
            currMoviePlots = self.story[currMovieKey]
            for currPlot in currMoviePlots:
                for word in currPlot.split():
                    vocabulary.add(word)
        return vocabulary

    def termFrequency(self, word, plots):
        """ 
        Returns term frequency score
        for a given word in plots
        """
        totalWordsInPlots = 0.0
        count = 0.0
        for currPlot in plots:
            totalWordsInPlots += len(currPlot.split())
            for currWord in currPlot.split():
                if word.lower() == currWord.lower():
                    count += 1.0
        return count
        '''
        TODO: Replace with logger class
        if (count > 0.0):
            # Make sure to normalize the TF score
            print 'TermFrequency: count: TotalWords: tfScore:'
            print '{0:.15f}'.format(count/totalWordsInDoc * self.idf(word))
        '''

    def inverseDocumentFrequency(self, word):
        """
        Returns the inverse document freq.
        of the given word that is calculated
        from all stories.
        """
        count = 0.0
        for currMovieKey in self.story:
            if word in self.movieToVocabularySet[currMovieKey]:
                count += 1.0
        if count == 0.0:
            # returns 0.0 if it doesn't exist from the vocabulary
            return count
        return math.log(len(self.story)/count, 2)

    def getSentenceVector(self, movieKey, sentence):
        sentenceVec = {}
        for currWord in sentence.split():
            if (currWord, movieKey) in self.tfIdfMatrix.keys():
                sentenceVec[currWord] = self.tfIdfMatrix[currWord, movieKey]
        return sentenceVec

    '''
    # Use tfidf for fetching the most relevant document from documents
    # Returns most probable document
    def tfidfRetrieval(self, word):
        maxScore = 0.0
        maxDoc = self.documents[0]
        # IDF score will be global for a word (independent of each individual documents)
        idfWord = self.idf(word)
        for currDoc in self.documents:
            currScore = self.termFrequency(word, currDoc) * idfWord
            if currScore >= maxScore:
                maxDoc = currDoc
                maxScore = currScore
        return maxDoc
    '''

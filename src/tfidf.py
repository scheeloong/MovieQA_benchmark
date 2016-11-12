import math
import numpy as np
import tensorflow as tf
import pickle
from src.movietokenizer import MovieTokenizer
from src.sparsematrix import SparseMatrix
# TODO: Replace all instances of shouldLog with a proper logger
shouldLog = False

class TfIdf(object):
    """
    This class implements the
    term frequency-inverse document frequency
    algorithm that is used for word embedding
    It serializes the trained words using pickle if it isn't already trained
    and deserialize the trained words from pickle"""
    
    # It doesn't normalize the TFIDF score as they are rank invariant

    # TODO: Implement serializing, checking if serialized to skip serialization
    # TODO: Implement deserializing
    def __init__(self, story, loadFile=None):
        """
        Story are the list of all available plots for the movie.
        Each plot contains a number of
        """

        # Tokenize alphanumeric with dash, lowercase, stem.  
        self.tokenizer = MovieTokenizer("[\w]+")

        # All the stories 
        #   Format is:
        #       [keyForStory, allPlotsForStory]
        self.story = story

        # A set for all possible vocabulary
        self.vocabularySet = set()
        # A map for each movie to it's vocabulary set
        # Used by IDF score calculation
        self.movieToVocabularySet = {}

        # Initialize both vocabularySet and movieVocabularySet
        self.initVocabulary()

        # The number of stories in this class
        self.numberOfStories = len(self.story)
        self.numberOfWords = len(self.vocabularySet)
        print 'Number of stories: ' + str(self.numberOfStories)
        print 'Number of Words: ' + str(self.numberOfWords)
        # A sparse matrix for tfIdfMatrix
        self.tfIdfMatrix = SparseMatrix(self.vocabularySet, self.story)

        self.idfVec = np.zeros(self.numberOfWords)
        for currWord in self.vocabularySet:
            self.idfVec[self.tfIdfMatrix.getWordIndex(currWord)] = self.inverseDocumentFrequency(currWord)

        # Can remove movieToVocabularySet since no longer need it after this
        self.movieToVocabularySet = {}

        self.initTfIdfMatrix(0.0)
        #self.initTfIdfMatrix(0.001)

        ''''
        # Get the tfIdfMatrix
        if loadFile is None:
            self.tfIdfMatrix = self.getTfIdfMatrix(0.001)
        if shouldLog:
            print self.tfIdfMatrix
        '''

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
        '''
        fileName = self.saveTfIdfMatrix(tfIdfThreshold)
        return self.loadTfIdfMatrix(fileName)
        '''
        self.initTfIdfMatrix(tfIdfThreshold)

    def loadTfIdfMatrix(self, fileName):
        """ 
        This method loads the already processed tfIdfMatrix using pickle.
        """
        with open(fileName, 'rb') as fp:
            return pickle.load(fp)

    def initTfIdfMatrix(self, tfIdfThreshold):
        """
        This method trains an entire new tfIdfMatrix and serializes it to file.
        """
        limitVocabForTesting = 15000 # TODO: Set this as a parameter
        # count = 0
        for currMovieKey in self.story:
            # print str(count) + ". Training movie: " + str(currMovieKey)
            # count += 1
            countOfWords = self.tokenizer.tokenizeDuplicatePerSentence(self.story[currMovieKey])
            for word in countOfWords:
                tfScore = np.log10(countOfWords[word]) + 1.0
                idfScore = self.idfVec[self.tfIdfMatrix.getWordIndex(word)]
                tfIdfScore = tfScore * idfScore
                '''
                if shouldLog:
                    print "tfscore: " + str(tfScore)
                    print "idfscore: " + str(idfScore)
                    print "tfidfscore: " + str(tfIdfScore)
                '''
                if tfIdfScore >= tfIdfThreshold:
                    self.tfIdfMatrix.setScore(word, currMovieKey, tfIdfScore)
        '''
        OBSOLETE as it takes less than 20  seconds to train now, don't need to waste time saving
        with  open('tfIdfMatrix.obj', 'wb') as fp:
            pickle.dump(self.tfIdfMatrix, fp)
        return 'tfIdfMatrix.obj'
        '''

    def initVocabulary(self):
        """
        Returns all the distinct words from all movies as a set
        """
        for currMovieKey in self.story:
            currMoviePlots = self.story[currMovieKey]
            self.movieToVocabularySet[currMovieKey] = set()
            for currPlot in currMoviePlots:
                self.movieToVocabularySet[currMovieKey].update(self.tokenizer.tokenizeAlphanumericLower(currPlot))
            self.vocabularySet.update(self.movieToVocabularySet[currMovieKey])

    '''
    # TODO: Replace this with python Collection counter class to count much faster Counter
    #       BUT MAYBE NOT CAUSE EVERY MOVIE LOOPS THROUGH SAME WORDS AGAIN
    #       A FASTER APPROACH MIGHT BE TO STORE MORE MEMORY
    #       EXAMPLE, STORE COUNTER CLASS FOR EVERY MOVIE
    #       STORE INVERTED INDEX FROM WORDS TO MOVIES THEY APPEAR IN
    #       SINCE MORE WORDS THAN MOVIES
    #       THINK ABOUT THIS ON PAPER THEN IMPLEMENT, DON'T IMPLEMENT BLINDLY
    #       FASTEST WAY SHOULD BE STORING ALL COUNTERS FOR EACH MOVIE
    #       STORING ALL IDF FOR EACH WORD LIKE YOU HAVE DONE
    #       THEN ITERATE THROUGH EVERY EVERY MOVIE and EVERY WORD ONCE
    #       since you need to go through VxM anyway to fill up matrix
    #       iterate the way that utilizes caches the best
    #       VxM = 12000 * 300 => 3600000 Million iterations, which may or may not be big

    #       LOL, YOU ARE LITERALLY ITERATING A DOCUMENT FOR EVERYWORD
    #       INSTEAD OF COUNTING AS YOU ITERATE THE DOCUMENT ONCE.


    #       Space memory analysis
            IdfVec is 12k * 1 => 12k memory
            Counter would be 12k * 293 => Large memory
            Thus, only store counter for each movie for each iteration => 12k * 1 memory and override
            And you wouldn't need to read it again after each movie so can override
            This is basically dynamic programming =) 
    '''

    def termFrequency(self, word, plots):
        """ 
        OBSOLETE
        Returns term frequency score
        for a given word in plots
        """
        count = 0.0
        for currPlot in plots:
            # TODO: Figure out if this is better or if previous approach of
            # number of plots word appears in is better
            count += self.tokenizer.countOccurence(currPlot, word)
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
        return np.log10(len(self.story)/count)

    def getSentenceVector(self, movieKey, sentence):
        sentenceVec = np.zeros(self.numberOfWords)
        cleanSentence = self.tokenizer.tokenizeAlphanumericLower(sentence)
        for currWord in cleanSentence:
            # Only values if it exist in the sparseMatrix
            if self.tfIdfMatrix.contains(currWord, movieKey):
                sentenceVec[self.tfIdfMatrix.getWordIndex(currWord)] = self.tfIdfMatrix.getScore(currWord, movieKey)
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

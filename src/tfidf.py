import MovieQA

'''
from data_loader import DataLoader
'''

import tensorflow as tf
import math

# To represent vectors
import numpy as np

shouldLog = True

class TfIdf(object):
    """
    This class implements the
    term frequency-inverse
    document frequency algorithm"""

    def __init__(self, story):
        """
        Documents are the list of all available plots for the movie.
        Each plot contains a number of
        """
        # All the stories 
        # [keyForStory, allPlotsForStory]
        self.story = story
        self.numberOfStories = len(self.story)
        # A set for all the possible vocabulary size
        self.vocabularySet = self.getVocabulary()
        # A map for each movie to it's vocabulary set to be used
        # by IDF score so don't have to calculate each time
        self.movieToVocabularySet = {}
        for currMovieKey in self.story:
            self.movieToVocabularySet[currMovieKey] = self.getVocabularyForMovie(currMovieKey)
        self.numberOfWords = len(self.vocabularySet)
        print 'Number of stories: ' + str(self.numberOfStories)
        print 'Number of Words: ' + str(self.numberOfWords)
        # arr[words][movieId]
        self.tfIdfMatrix = self.getTfIdfMatrix(0.001)
        # The matrix dimension is (|Vocabulary Size| * numberOfStories)
        # But excluded those with tfidf score below a certain threshold
        """
                                movieId_1       ...    movieId_numberOfStories
                                ____________|_______|________________________
        word_1             | tfidf(word_1, movieId_1)            |       |  
        ...                |                |       |
        word_numberOfWords |                |       |
        """
        if shouldLog:
            print self.tfIdfMatrix

    def getTfIdfMatrix(self, tfIdfThreshold):
        tfIdfMatrix = {}
        # TODO: Clean up words with nltk (get rid of ',' and match similar words)
        count = 0
        limitVocabForTesting = 20000
        for currWord in self.vocabularySet:
            if shouldLog:
                print "makingMatrix for word: " + currWord
            exist = False
            if count > limitVocabForTesting:
                break
            idfScore = self.inverseDocumentFrequency(currWord)
            # Skip this word if it is useless ('the')
            if (idfScore < tfIdfThreshold):
                continue
            # Calculate the idf score for every word
            for currMovieKey in self.story:
                # Calculate the tfScore for this word in this movie
                tfScore = self.termFrequency(currWord, self.story[currMovieKey])
                if tfScore == 0:
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
        return tfIdfMatrix

    # Used by IDF score
    def getVocabularyForMovie(self, currMovieKey):
        vocabulary = set()
        currMoviePlots = story[currMovieKey]
        for currPlot in currMoviePlots:
            for word in currPlot.split():
                vocabulary.add(word)
        return vocabulary

    def getVocabulary(self):
        vocabulary = set()
        for currMovieKey in story:
            currMoviePlots = story[currMovieKey]
            for currPlot in currMoviePlots:
                for word in currPlot.split():
                    vocabulary.add(word)
        return vocabulary

    def termFrequency(self, word, plots):
        """ 
        Returns score for given word in plots for a given movie
            '''
            if (count > 0.0 ):
                # Make sure to normalize the TF score
                print 'TermFrequency: count: TotalWords: tfScore:'
                print '{0:.15f}'.format(count/totalWordsInDoc * self.idf(word))
            '''
        """
        totalWordsInPlots = 0.0
        count = 0.0
        for currPlot in plots:
            totalWordsInPlots += len(currPlot.split())
            for currWord in currPlot.split():
                # Make sure it's casing independent.
                if word.lower() == currWord.lower():
                    count += 1.0
        return count

    def inverseDocumentFrequency(self, word):
        """
        Returns the inverse document freq.
        of the given word in all documents)
        """
        count = 0.0
        for currMovieKey in self.story:
            if word in self.movieToVocabularySet[currMovieKey]:
                count += 1.0
        # Doesn't work if count = 0 (word is no where inside vocabulary)
        if count != 0.0:
            return math.log(len(self.story)/count, 2)
        return 0.0

    def getSentenceVector(self, movieKey, sentence):
        sentenceVec = {}
        for currWord in sentence.split():
            if (currWord, movieKey) in self.tfIdfMatrix.keys():
                sentenceVec[currWord] = self.tfIdfMatrix[currWord, movieKey]
        return sentenceVec

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

def cosineSimilarity(vecA, vecB):
    # Cosine similarity
    sumA = 0.0
    for keyA in vecA:
        sumA += math.pow(vecA[keyA], 2)
    normA = math.sqrt(sumA)

    sumB = 0.0
    for keyB in vecB:
        sumB += math.pow(vecB[keyB], 2)
    normB = math.sqrt(sumB)
    if normA == 0.0:
        return 0.0
    if normB == 0.0:
        return 0.0
    sumDot = 0.0
    for keyAB in vecA:
        if keyAB in vecB.keys():
            sumDot += vecA[keyAB] * vecB[keyAB]
    cosineSimilarity = sumDot / (normA * normB)
    return cosineSimilarity
#---------------------------------------------------------------------------------------------
if __name__ == "__main__":
#---------------------------------------------------------------------------------------------
    dL = MovieQA.DataLoader()
    # Use training data for training
    [story, qa]  = dL.get_story_qa_data('train', 'plot')
    # Use test data for testing
    [story2, qa2]  = dL.get_story_qa_data('test', 'plot')
    # TODO: Uncomment this once done questions
    tfidf_ = TfIdf(story)

    # Run validation test
    numChoices = [0, 0, 0, 0, 0]
    numQuestions = len(qa)
    numCorrect = 0
    questionCount = 0
    for currQA in qa:
        questionCount += 1
        QuestionVec = tfidf_.getSentenceVector(currQA.imdb_key, currQA.question)
        count = 0
        choice = -1
        bestPlotVec = story[currQA.imdb_key][0]
        currPlotScore = -1
        for currPlot in story[currQA.imdb_key]:
            PlotVec = tfidf_.getSentenceVector(currQA.imdb_key, currPlot)
            score = cosineSimilarity(QuestionVec, PlotVec)
            if  score > currPlotScore:
                currPlotScore =  score
                bestPlotVec = PlotVec
        if currPlotScore == -1:
            print "ERROR: Should have a Plot"
        currQAScore = -1.0
        for answer in currQA.answers:
            AnswerVec = tfidf_.getSentenceVector(currQA.imdb_key, answer) 
            #score = cosineSimilarity(QuestionVec, AnswerVec)
            score = cosineSimilarity(bestPlotVec, AnswerVec)
            if  score > currQAScore:
                currQAScore = score
                choice = count
            count += 1
        if choice == -1:
            if shouldLog:
                print "ERROR: Should have an answer"
            continue
        numChoices[choice] += 1
        if choice == currQA.correct_index:
            numCorrect += 1
    print 'numCorrect: ' + str(numCorrect)
    print 'numQuestions: ' + str(numQuestions)
    print numChoices

    # Information Retrieval component
    #doc = tfidf_.tfidfRetrieval("narration")

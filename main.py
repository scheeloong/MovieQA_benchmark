import MovieQA
from src.tfidf import TfIdf

from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

import cProfile # Profiling 
import datetime
import nltk # Tokenization with Regular Expression, Stemming
import math # abs
import numpy as np
import tensorflow as tf
import logging # TODO: Replace all print with logging class
# TODO: Perform Cross Validation 
# TODO: Performing profiling across code once done optimizing everything you can

# This function runs the tfidf on the test methods
def runTfIdf(trainPlots, testQuestions):
    # FIXME: Class relies on trainPlots to initialize class variables
    #       which must be consistent with file passed in
    # TODO: Compile if it doesn't exist yet, skip compilation if already stored data.
    #tfidf_ = TfIdf(trainPlots, 'tfIdfMatrix.obj')
    startTime = datetime.datetime.now()
    tfidf_ = TfIdf(trainPlots)
    endTime = datetime.datetime.now()
    # Run validation test
    numChoices = np.array([0, 0, 0, 0, 0])
    corrChoices = np.array([0, 0, 0, 0, 0])
    numQuestions = len(testQuestions)
    finalAnswers = np.zeros(numQuestions)
    numCorrect = 0
    currQaNum = 0
    for currQA in testQuestions:
        currQaNum += 1
        currTotalScore = -1.0
        choice = -1
        QuestionVec = tfidf_.getSentenceVector(currQA.imdb_key, currQA.question)
        AnsA = tfidf_.getSentenceVector(currQA.imdb_key, currQA.answers[0])
        AnsB = tfidf_.getSentenceVector(currQA.imdb_key, currQA.answers[1])
        AnsC = tfidf_.getSentenceVector(currQA.imdb_key, currQA.answers[2])
        AnsD = tfidf_.getSentenceVector(currQA.imdb_key, currQA.answers[3])
        AnsE = tfidf_.getSentenceVector(currQA.imdb_key, currQA.answers[4])
        # TODO: Implement window for plots 
        for currPlot in trainPlots[currQA.imdb_key]:
            PlotVec = tfidf_.getSentenceVector(currQA.imdb_key, currPlot)
            A = np.array([PlotVec, QuestionVec, AnsA, AnsB, AnsC, AnsD, AnsE])
            ASparse = sparse.csr_matrix(A)
            scores = cosine_similarity(ASparse)
            # TODO: Make into function
            plotQuestionScore = scores[0, 1]
            answerScores = np.array([scores[0,2], scores[0,3], scores[0,4], scores[0,5], scores[0,6]])
            plotAnswerScore = max(answerScores)
            score = (plotQuestionScore + plotAnswerScore)/2.0
            if  score > currTotalScore:
                currTotalScore = score
                choice = np.argwhere(answerScores == max(answerScores))[0,0]
        if currTotalScore == -1.0:
            print "ERROR: Should have a total score"
            continue
        numChoices[choice] += 1
        corrChoices[currQA.correct_index] += 1
        if choice == currQA.correct_index:
            numCorrect += 1
        finalAnswers[currQaNum-1] = choice
        '''
        if not (currQaNum % 1000):
            print numChoices
            print corrChoices
            print 'numCorrect: ' + str(numCorrect) 
            print 'Accurary: ' + str((numCorrect*1.0)/currQaNum)
            print 'currQaNum: ' + str(currQaNum)
            print 'NumQuestions: ' + str(numQuestions)
        '''
    print 'numCorrect: ' + str(numCorrect)
    print 'numQuestions: ' + str(numQuestions)
    print 'chosen Answers: ' + str(numChoices)
    print 'correct Answers: ' + str(corrChoices)
    print 'currQaNum: ' + str(currQaNum)
    print 'Accurary: ' + str((numCorrect*1.0)/numQuestions)
    print 'StartTrainingTime: ' + str(startTime.time())
    print 'EndTrainingTime: ' + str(endTime.time())
    elapsedTime = endTime - startTime
    hours, remainder = divmod(elapsedTime.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    totalDays = elapsedTime.days
    print 'Training Time: Days: ' + str(totalDays) +  " hours: " + str(hours) + ' minutes: ' + str(minutes) +  ' seconds: ' + str(seconds)
    return finalAnswers
    
    
#---------------------------------------------------------------------------------------------
if __name__ == "__main__":
    startTime = datetime.datetime.now()
    dL = MovieQA.DataLoader()
    # Use training data for training
    [storyTrain, qaTrain]  = dL.get_story_qa_data('train', 'plot')
    # Use test data for testing
    [storyTest, qaTest]  = dL.get_story_qa_data('test', 'plot')
    # TODO: FIgure out how to run qaTest if the movies are totally different 
    #       from qaTrain which means you can't fetch the sentence vectors for it.
    finalAnswers = runTfIdf(storyTrain, qaTrain)
    endTime = datetime.datetime.now()
    print 'TotalStartTime: ' + str(startTime.time())
    print 'TotalEndTime: ' + str(endTime.time())
    elapsedTime = endTime - startTime
    hours, remainder = divmod(elapsedTime.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    totalDays = elapsedTime.days
    print 'Total Time Taken: Days: ' + str(totalDays) +  " hours: " + str(hours) + ' minutes: ' + str(minutes) +  ' seconds: ' + str(seconds)

    print 'Answers Chosen'
    count = 0
    for val in finalAnswers:
        print 'test:' + str(count) + ' ' + str(int(val))
        count += 1
#---------------------------------------------------------------------------------------------

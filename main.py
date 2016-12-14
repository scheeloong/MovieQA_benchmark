import MovieQA
from src.tfidf import TfIdf
from src.htmloutput import HtmlOutput

import datetime
import logging
import math
import nltk
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import tensorflow as tf

# TODO: Perform Cross Validation 
# TODO: Plot a histogram for the scores and see how the graph
#       changes when you change the algorithm or parameters. 

def runTfIdf(trainPlots, testQuestions):
    # To output results in a beautified html file
    correctHtml = HtmlOutput("correctfile.html")
    wrongHtml = HtmlOutput("wrongfile.html")

    startTime = datetime.datetime.now()

    # Train the plots with TFIDF score
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

        # Sentence vectors for question and the 5 possible answers
        QuestionVec = tfidf_.getSentenceVector(currQA.imdb_key, currQA.question)
        AnsA = tfidf_.getSentenceVector(currQA.imdb_key, currQA.answers[0])
        AnsB = tfidf_.getSentenceVector(currQA.imdb_key, currQA.answers[1])
        AnsC = tfidf_.getSentenceVector(currQA.imdb_key, currQA.answers[2])
        AnsD = tfidf_.getSentenceVector(currQA.imdb_key, currQA.answers[3])
        AnsE = tfidf_.getSentenceVector(currQA.imdb_key, currQA.answers[4])

        chosenPlot = ""
        for currPlot in trainPlots[currQA.imdb_key]:
            PlotVec = tfidf_.getSentenceVector(currQA.imdb_key, currPlot)
            matrix = np.array([PlotVec, QuestionVec, AnsA, AnsB, AnsC, AnsD, AnsE])
            matrixSparse = sparse.csr_matrix(matrix)

            # Calculates scores using cosine similarity
            scores = cosine_similarity(matrixSparse)
            plotQuestionScore = scores[0, 1]
            answerScores = np.array([scores[0,2], scores[0,3],
                           scores[0,4], scores[0,5], scores[0,6]])
            plotAnswerScore = max(answerScores)
            score = (plotQuestionScore + plotAnswerScore)/2.0

            # Find plot with maximum score
            if  score > currTotalScore:
                currTotalScore = score
                choice = np.argwhere(answerScores == max(answerScores))[0,0]
                chosenPlot = currPlot

        # Update correct and chosen answer choices
        numChoices[choice] += 1
        corrChoices[currQA.correct_index] += 1

        # Plot output to html files
        if choice == currQA.correct_index:
            numCorrect += 1
            correctHtml.formTable(currQA, chosenPlot, choice, currTotalScore)
        else:
            wrongHtml.formTable(currQA, chosenPlot, choice, currTotalScore)
        finalAnswers[currQaNum-1] = choice

    logTestPeformance(numQuestions, numChoices, corrChoices, numCorrect)
    logTimeInfo(startTime, endTime, "Training Time")

    correctHtml.close() 
    wrongHtml.close()
    return finalAnswers

def logTestPeformance(numQuestions, numChoices, corrChoices, numCorrect):
    ''' Logs the test accuracy performances '''
    logging.info('Number of Questions: ' + str(numQuestions))
    logging.info('Correct Answer Choices: ' + str(corrChoices))
    logging.info('Chosen Answer Choices: ' + str(numChoices))
    logging.info('Number of Correct Answers: ' + str(numCorrect))
    logging.info('Accurary: ' + str((numCorrect*1.0)/numQuestions))

def logElapsedTime(elapsedTime, message):
    ''' Logs the elapsedTime with a given message '''
    hours, remainder = divmod(elapsedTime.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    totalDays = elapsedTime.days
    logging.info(str(message) + ': Days: ' + str(totalDays) +  " hours: " + str(hours) + ' minutes: ' + str(minutes) +  ' seconds: ' + str(seconds))

def logTimeInfo(startTime, endTime, message):
    ''' Logs information about elapsedTime '''
    elapsedTime = endTime - startTime
    logElapsedTime(elapsedTime, message)
    
if __name__ == "__main__":
    logging.basicConfig(filename='output.log', level=logging.DEBUG)
    startTime = datetime.datetime.now()

    # Download the data for training and testing.
    dL = MovieQA.DataLoader()
    [storyTrain, qaTrain]  = dL.get_story_qa_data('train', 'plot')

    # Run the TFIDF algoritm
    finalAnswers = runTfIdf(storyTrain, qaTrain)

    endTime = datetime.datetime.now()

    logTimeInfo(startTime, endTime, "Total Time")

    # Write the chosen answers from the algorithm
    # in the format that is used for submission to MovieQA leaderboard.
    file = open('testResults.txt', "w")
    file.write('Answers Chosen')
    count = 0
    for answerPicked in finalAnswers:
        file.write('test:' + str(count) + ' ' + str(int(answerPicked)))
        count += 1
    file.close()

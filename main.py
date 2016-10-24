import MovieQA
from src.tfidf import TfIdf

import tensorflow as tf
import math

# To represent vectors
import numpy as np

# This function runs the tfidf on the test methods
def runTfIdf():
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


#---------------------------------------------------------------------------------------------
if __name__ == "__main__":
    runTfIdf()
#---------------------------------------------------------------------------------------------

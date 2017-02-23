import MovieQA
from src.tfidf import TfIdf
from src.htmloutput import HtmlOutput 
import datetime
import logging
import math
import nltk
import numpy as np
from w2vtester import W2VTester 
from util import log_time_info

class FusionTester(object):
    def __init__(self, extension='plt'):
        mqa = MovieQA.DataLoader()
        self.story_raw, self.qa = mqa.get_story_qa_data('train', 'plot')

        # Train the plots with TFIDF score
        self.tfidf = TfIdf(self.story_raw)
        self.w2vtester = W2VTester(self.story_raw, self.qa)

    def normalize(self, mat):
        if mat.ndim == 1:
            return mat/(np.linalg.norm(mat)+1e-6)
        return mat/(np.linalg.norm(mat, axis=1, keepdims=True)+1e-6)

    def score_tfidf(self, q):
        # Sentence vectors for question, answers, plot
        question_vec = self.tfidf.getSentenceVector(q.imdb_key, q.question)
        answer_matrix = self.tfidf.getSentenceVectors(q.imdb_key, q.answers)
        plot_matrix = self.tfidf.getSentenceVectors(q.imdb_key, self.story_raw[q.imdb_key])

        qscore = plot_matrix.dot(question_vec).reshape(-1,1)
        ascore = plot_matrix.dot(answer_matrix.T)
        score = ascore + qscore
        return score

    def predict_tfidf(self, q):
        score = self.score_tfidf(q)
        prediction = np.unravel_index(score.argmax(), score.shape)
        return prediction[1], prediction[0], score[prediction]

    @log_time_info
    def test(self, weight):
        ''' Run w2v on the plots.
        '''

        # Start testing.
        nCorrect = 0.0
        nCorrectSame = 0.0
        prediction_distribution = [0,0,0,0,0]
        correct_distribution = [0,0,0,0,0]
        nTried = len(self.qa)

        for q in self.qa:
            tfidf_scores = np.amax(self.score_tfidf(q), axis=0)
            w2v_scores = np.amax(self.w2vtester.score(q), axis=0)
            scores = weight*self.normalize(tfidf_scores) + self.normalize(w2v_scores)
            prediction = scores.argmax()

            if prediction == q.correct_index:
                nCorrect+=1
            prediction_distribution[prediction] +=1
            correct_distribution[q.correct_index] +=1


        print("TFIDF Accuracy:", nCorrect/nTried)
        print("Percentage Overlap in Correct:", nCorrectSame/nCorrect)
        print("Predicted Answers: ", prediction_distribution)
        print("Correct Answers: ",correct_distribution)
        return(nCorrect/nTried)

if __name__ == "__main__":
    fusion_tester = FusionTester()
    weights =  [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    weights_2 = [2.5, 5, 7.5, 12, 15, 17, 20]
    weights_3 = [20, 35, 50, 80, 120, 300, 500]
    accuracies = []
    for w in weights_2:
        print "---------------------- %f:1 tfidf:w2v ratio -------------------------" % w
        accuracy = fusion_tester.test(w)
        print "---------------------------------------------------------------------"
        accuracies.append(accuracy)

    print weights
    print accuracies
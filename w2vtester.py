import MovieQA
import numpy as np
from w2v.word2vec import Word2Vec
import sys

class W2VTester(object):

    def __init__(self, story_raw, qa, extension='plt', postprocess=True):
        self.qa = qa
        self.w2v = Word2Vec(extension, postprocess=postprocess)
        self.maxSentenceLength = 0

        # Build each plot into a matrix of sentence embeddings.
        self.story_matrices = {}
        for imdb_key in story_raw:
            self.story_matrices[imdb_key] = self.w2v.get_vectors_for_raw_text(story_raw[
            
                                                                              imdb_key])

            #print 'raw plot'
            #print story_raw[imdb_key]
            # print 'processed plot'
            #print self.story_matrices[imdb_key]
            print self.story_matrices[imdb_key].shape
            self.maxSentenceLength = max(self.maxSentenceLength, self.story_matrices[imdb_key].shape[0])
            # sys.exit(0)
        print self.maxSentenceLength

    def score(self, q):
        ''' Make a prediction for the QA.
        '''
        # Process question and answers.
        question = self.w2v.get_sentence_vector(q.question)
        answers = self.w2v.get_text_vectors(q.answers)
        #print "ori question"
        #print q.question
        print "parsed question"
        #print question
        print question.shape
        #print "ori answer"
        #print q.answers
        print "parsed answer"
        #print answers
        print answers.shape

        # Calculate similarity
        qscore = self.story_matrices[q.imdb_key].dot(question).reshape(-1, 1)
        ascore = self.story_matrices[q.imdb_key].dot(answers.T)
        score = ascore + qscore
        return score

    def predict(self, q):
        scores = self.score(q)
        # Prediction is (plot line#, answer#)
        prediction = np.unravel_index(scores.argmax(), scores.shape)
        # print scores.shape
        # prediction = scores.argmax()
        # print prediction
        # return prediction, "", scores[prediction]
        return prediction[1], prediction[0], scores[prediction]

    def test(self):
        ''' Run w2v on the plots.
        '''

        # Start testing.
        nCorrect = 0.0
        nTried = len(self.qa)

        for q in self.qa:
            prediction, reference, confidence = self.predict(q)
            if prediction == q.correct_index:
                nCorrect += 1

        print("Word2Vec Accuracy:", nCorrect / nTried)


if __name__ == "__main__":
    mqa = MovieQA.DataLoader()
    print "-----Training Data-----"
    story_raw, qa = mqa.get_story_qa_data('train', 'plot')
    w2v_tester = W2VTester(story_raw, qa)
    w2v_tester.test()

    print "-----Validation Data-----"
    story_raw, qa = mqa.get_story_qa_data('val', 'plot')
    w2v_tester = W2VTester(story_raw, qa)
    w2v_tester.test()

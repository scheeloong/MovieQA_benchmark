import MovieQA
import numpy as np
from w2v.word2vec import Word2Vec


class W2VTester(object):
    def __init__(self, extension='plt'):
        mqa = MovieQA.DataLoader()
        story_raw, self.qa = mqa.get_story_qa_data('train', 'plot')
        self.w2v = Word2Vec(extension)

        # Build each plot into a matrix of sentence embeddings.
        self.story_matrices = {}
        for imdb_key in story_raw:
            self.story_matrices[imdb_key] = self.w2v.get_vectors_for_raw_text(story_raw[imdb_key])

    def predict(self, q):
        # Process question and answers.
        question = self.w2v.get_sentence_vector(q.question) 
        answers = self.w2v.get_text_vectors(q.answers)  

        # Calculate similarity
        qscore = self.story_matrices[q.imdb_key].dot(question).reshape(-1, 1)
        ascore = self.story_matrices[q.imdb_key].dot(answers.T)
        score = ascore + qscore
        # Prediction is (plot line#, answer#)
        prediction = np.unravel_index(score.argmax(), score.shape)
        return prediction[1], prediction[0], score[prediction]

    def test(self):
        ''' Run w2v on the plots.
        '''
        
        # Start testing.
        nCorrect = 0.0
        nTried = len(self.qa)

        for q in self.qa:
            prediction, reference, confidence = self.predict(q)
            if prediction == q.correct_index:
                nCorrect+=1

        
        print("Word2Vec Accuracy:", nCorrect/nTried)


if __name__ == "__main__":
    w2v_tester = W2VTester()
    w2v_tester.test()
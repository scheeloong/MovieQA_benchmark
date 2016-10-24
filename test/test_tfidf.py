""" 
Test for Term Frequency Inverse Document Frequency
TODO(scheeloong): Implement test
"""
import unittest

# Import the package (which is made by having a file called __init__.py
import src
import MovieQA
# Import the module tfidf.py
from src import tfidf
# From tfidf.py, import the class TfIdf
from src.tfidf import TfIdf

class TestTfIdf(unittest.TestCase):
    def test_nothing(self):
        self.assertEqual('lala', 'lala')
        dL = MovieQA.DataLoader()
        # Use training data for training
        [story, qa]  = dL.get_story_qa_data('train', 'plot')
        # Use test data for testing
        [story2, qa2]  = dL.get_story_qa_data('test', 'plot')
        # TODO: Uncomment this once done questions
        tfidf_ = TfIdf(story)

if __name__ == '__main__':
    unittest.main()

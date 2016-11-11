from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem import PorterStemmer

class MovieTokenizer(object):

    def __init__(self, tokenizeRegExp):
        self.tokenizer = RegexpTokenizer(tokenizeRegExp)
        self.ps = PorterStemmer()

    def countOccurence(self, sentences, word):
        count = 0.0
        for currWord in self.tokenizer.tokenize(sentences):
            if word == currWord:
                count += 1.0
        return count

    def tokenizeDuplicate(self, sentences):
        # Repeat the elements for counting
        vocabulary = {}
        for sentence in sentences:
            for word in self.tokenizer.tokenize(sentence):
                word = self.ps.stem(word.lower())
                if word not in vocabulary:
                    vocabulary[word] = 1
                else:
                    vocabulary[word] += 1
        return vocabulary

    def tokenizeAlphanumericLower(self, sentences):
        """
        Returns a set of unique words broken up after being tokenize
        by alphanumeric and lowerCase
        """
        vocabulary = set()
        for word in self.tokenizer.tokenize(sentences):
            vocabulary.add(self.ps.stem(word.lower()))
        return vocabulary


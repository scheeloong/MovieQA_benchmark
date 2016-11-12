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

    def tokenizeDuplicatePerSentence(self, sentences):
        # Repeat the elements for counting
        # but only one time for each sentence
        vocabulary = {}
        for sentence in sentences:
            # Use set to make sure every word is unique
            # within a sentence
            for word in set(self.tokenizer.tokenize(sentence)):
                word = self.ps.stem(word.lower())
                if word not in vocabulary:
                    vocabulary[word] = 1
                else:
                    vocabulary[word] += 1
        return vocabulary

    def tokenizeDuplicate(self, sentences):
        # Repeat the elements for counting 
        # included repeated times in the same sentence.
        vocabulary = {}
        for sentence in sentences:
            # Don't use set, can have repeated words per sentence
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


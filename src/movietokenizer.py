from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer

class MovieTokenizer(object):

    def __init__(self, tokenizeRegExp):
        ''' 
        Initialize tokenizer with given regular expression
        it uses the WordNetLemmatizer as it performs the best for this training data.
        '''
        self.tokenizer = RegexpTokenizer(tokenizeRegExp)
        self.lemmatizer = WordNetLemmatizer() # 47.5 on training set

    def countOccurence(self, sentences, word):
        ''' Counts occurence of work in sentence '''
        count = 0.0
        for currWord in self.tokenizer.tokenize(sentences):
            if word == currWord:
                count += 1.0
        return count

    def tokenizeDuplicatePerSentence(self, sentences):
        ''' Counts number of sentences that each word appears in '''
        vocabulary = {}
        for sentence in sentences:
            for uniqueWord in set(self.tokenizer.tokenize(sentence)):
                lemmatizedUniqueLowerWord = self.lemmatizer.lemmatize(uniqueWord.lower())
                if lemmatizedUniqueLowerWord not in vocabulary:
                    vocabulary[lemmatizedUniqueLowerWord] = 1
                else:
                    vocabulary[lemmatizedUniqueLowerWord] += 1
        return vocabulary

    def tokenizeDuplicate(self, sentences):
        '''
        Repeat the elements for counting
        if it is repeated in the same sentence.
        '''
        vocabulary = {}
        for sentence in sentences:
            # Don't use set, can have repeated words per sentence
            for word in self.tokenizer.tokenize(sentence):
                lemmatizedLowerWord = self.lemmatizer.lemmatize(word.lower())
                if word not in vocabulary:
                    vocabulary[lemmatizedLowerWord] = 1
                else:
                    vocabulary[lemmatizedLowerWord] += 1
        return vocabulary

    def tokenizeAlphanumericLower(self, sentences):
        """
        Returns a set of unique words broken up after being tokenize
        by alphanumeric and lowerCase
        """
        vocabulary = set()
        for word in self.tokenizer.tokenize(sentences):
            lemmatizedLowerWord = self.lemmatizer.lemmatize(word.lower())
            vocabulary.add(lemmatizedLowerWord)
        return vocabulary

    def tokenizeAlphanumericLowerList(self, sentences):
        """
        Returns a list of unique words broken up after being tokenize
        by alphanumeric and lowerCase
        """
        vocabulary = list()
        for word in self.tokenizer.tokenize(sentences):
            lemmatizedLowerWord = self.lemmatizer.lemmatize(word.lower())
            vocabulary.append(lemmatizedLowerWord)
        return vocabulary

    def tokenizedAlphanumericPairs(self, sentences):
        """
        Returns a list of unique words broken up after being tokenize
        by alphanumeric and lowerCase
        """
        vocabulary = list()
        for word in self.tokenizer.tokenize(sentences):
            lemmatizedLowerWord = self.lemmatizer.lemmatize(word.lower())
            vocabulary.append(lemmatizedLowerWord)
        return (self.tokenizer.tokenize(sentences), vocabulary)

    def tokenizeOrderedAlphanumericLower(self, sentences):
        """
        Returns a set of unique words broken up after being tokenize
        by alphanumeric and lowerCase
        """
        clean_sentences = []
        for word in self.tokenizer.tokenize(sentences):
            lemmatizedLowerWord = self.lemmatizer.lemmatize(word.lower())
            clean_sentences.append(lemmatizedLowerWord)
        return clean_sentences

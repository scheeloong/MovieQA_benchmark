from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer

# This is a temporary file edited from MovieTokenizer
# to not actually tokenize anything as it is use for analysis and comparison
# to the output with tokenization
class NoTokenizer(object):
# class MovieTokenizer(object):

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
        #for currWord in self.tokenizer.tokenize(sentences):
        for currWord in sentences.split():
            if word == currWord:
                count += 1.0
        return count

    def tokenizeDuplicatePerSentence(self, sentences):
        ''' Counts number of sentences that each word appears in '''
        vocabulary = {}
        for sentence in sentences:
            ##for lemmatizedUniqueLowerWord in set(self.tokenizer.tokenize(sentence)):
            for lemmatizedUniqueLowerWord in set(sentence.split()):
            #for uniqueWord in set(self.tokenizer.tokenize(sentence)):
                # lemmatizedUniqueLowerWord = self.lemmatizer.lemmatize(uniqueWord.lower())
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
            ##for word in self.tokenizer.tokenize(sentence):
            for word in sentence.split():
                lemmatizedLowerWord = word
                #lemmatizedLowerWord = self.lemmatizer.lemmatize(word.lower())
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
        ##for lemmatizedLowerWord in self.tokenizer.tokenize(sentences):
        for lemmatizedLowerWord in sentences.split():
         
        #for word in self.tokenizer.tokenize(sentences):
            #lemmatizedLowerWord = self.lemmatizer.lemmatize(word.lower())
            vocabulary.add(lemmatizedLowerWord)
        return vocabulary

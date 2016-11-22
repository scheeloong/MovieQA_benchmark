from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer, WordNetLemmatizer

class MovieTokenizer(object):

    def __init__(self, tokenizeRegExp):
        self.tokenizer = RegexpTokenizer(tokenizeRegExp)
        self.ps = PorterStemmer()
        self.ls = LancasterStemmer()
        self.ss = SnowballStemmer("english")
        #self.stemmer = self.ls # 46.2 on training set
        #self.stemmer = self.ps # 46.77 on training set
        self.stemmer = self.ss # 46.79 on training set
        self.lemmatizer = WordNetLemmatizer() # 47.5 on training set

    def usePorterStemmer(self):
        self.stemmer = self.ps

    def useLancasterStemmer(self):
        self.stemmer = self.ls

    def useSnowballStemmer(self):
        self.stemmer = self.ss

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
                word = word.lower()
                #word = self.stemmer.stem(word)
                word = self.lemmatizer.lemmatize(word)
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
                word = word.lower()
                #word = self.stemmer.stem(word)
                word = self.lemmatizer.lemmatize(word)
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
            word = word.lower()
            #word = self.stemmer.stem(word)
            word = self.lemmatizer.lemmatize(word)
            vocabulary.add(word)
            #vocabulary.add(self.stemmer.stem(word.lower()))
        return vocabulary


import MovieQA
from src.tfidf import TfIdf
from src.htmloutput import HtmlOutput 
from src.memn2nmovie import MemN2N
from w2v.word2vec import Word2Vec

import datetime
import logging
import math
import nltk
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import tensorflow as tf

# TODO: Perform Cross Validation 
# TODO: Plot a histogram for the scores and see how the graph
#       changes when you change the algorithm or parameters. 
def runMemN2N(trainPlots, testQuestions):
    memn2n = MemN2N(trainPlots, testQuestions)

class MemN2N(object):
    def __init__(self, story, qa, extension='plt', postprocess=True):
        '''
        -------------------------------------------------------
        b = self.batchSize = 9 => 32
        V = vocabulary size = 20 => HUGE
        z = numberOfSentence = 10
        l = numberOfWordsInSentence = 6
        d = memn2n word_embedding_size = 2 => anyNumber
        -------------------------------------------------------
        W  = (d, V) 
        story data (9, 10, 6) = (b, z, l)
        Embeddings A (20, 2) = (V, d)
        Embedding Lookup A (9, 10, 6, 2) = (b, z, l, d)
        Memory Matrix M (9, 10, 2) = (b, z, d)
        Question Data (9, 1, 6) = (b, 1, l)
        Control Signal U (9, 1, 2) = (b, 1, d)
        Memory Selection (9, 10, 1) = (b, z, 1)
        p (9, 10, 1) = (b, z, 1)
        c_set (9, 2, 10) = (b, d, z)
        o (9, 2) = (b, d, 1)
        o_u_sum = (b, 1, d)
        Correct Prediction (9, 1) = (b, 1)
        -------------------------------------------------------
        '''
        # w2v contains the embeddings to be trained

        # TODO: Temporary hard coded values below
        self.memorySize = 100
        self.batchSize = 32
        self.learningRate = 0.01
        self.w2v = Word2Vec(extension, postprocess=postprocess)
        self.qa = qa # all the questions
        self.numEpoch = 1

        # Parse the values
        self.word2VecDim = 0  # initialize
        self.embedDim = 0 #initialize
        self.maxNumSentences = 0  # initialize
        self.numAnswers = 0 # initialize
        self.numQuestions = 0 # initialize
        # Embeddings for the story matrices
        self.storyMatrices= {} # storyMatrices indexed by movieKey

        for movieKey in story:
            self.storyMatrices[movieKey] = self.w2v.get_vectors_for_raw_text(story[movieKey])
            self.maxNumSentences = max(self.maxNumSentences, self.storyMatrices[movieKey].shape[0])
            # TODO: Convert all story data to max numsentences

        for movieKey in story:
            self.word2VecDim = self.storyMatrices[movieKey].shape[1]
            self.embedDim = self.word2VecDim # TODO: Confirm if its okay to set as the same
            break

        for currQA in self.qa:
            answers = self.w2v.get_text_vectors(currQA.answers) #print "ori question"
            self.numAnswers = answers.shape[0] 
            break;
        for currQA in self.qa:
            self.numQuestions += 1
        for currQA in self.qa:
            question = self.w2v.get_sentence_vector(currQA.question)
            answers = self.w2v.get_text_vectors(currQA.answers) 
        self.vocabularySize = self.w2v.getNumVocabulary()

        # Form the 3 input matrices
        self.X = np.zeros((self.numQuestions, self.maxNumSentences, self.word2VecDim))
        self.q = np.zeros((self.numQuestions, self.word2VecDim))
        self.a = np.zeros((self.numQuestions, self.numAnswers, self.word2VecDim))
        numQ = 0
        for currQA in self.qa:
            question = self.w2v.get_sentence_vector(currQA.question)
            answers = self.w2v.get_text_vectors(currQA.answers) 
            self.q[numQ] = question
            self.a[numQ] = answers
            if self.storyMatrices[currQA.imdb_key].shape[0] != self.maxNumSentences:
                difference = self.maxNumSentences - self.storyMatrices[currQA.imdb_key].shape[0]
                temp = self.storyMatrices[currQA.imdb_key]
                haha = np.zeros((difference, self.word2VecDim))
                temp = np.vstack((temp, haha))
                self.storyMatrices[currQA.imdb_key] = temp
            self.X[numQ] = self.storyMatrices[currQA.imdb_key]
            numQ += 1

        print "maxNumSentence", self.maxNumSentences
        print "word2VecDim", self.word2VecDim
        print "numAnswers", self.numAnswers
        print "vocabularySize", self.vocabularySize
        print "numQuestions", self.numQuestions
        # self.sentenceLength = 99999 # TODO: Confirm if dont need this
        print "Done Parsing for MemN2N"

        self.buildGraphRunSess()

    def buildGraphRunSess(self):
        #Create tensorflow graph
        graph = tf.Graph()

        with graph.as_default(), tf.device('/cpu:0'):
            # TODO: Figure out if the sizes below are right
            #story_data = tf.placeholder(tf.float32, shape=[None, self.maxNumSentences, self.vocabularySize], name="storydata")
            story_data = tf.placeholder(tf.float32, shape=[None, self.maxNumSentences, self.embedDim], name="storydata")
            question_data = tf.placeholder(tf.float32, shape=[None, 1, self.embedDim], name="questiondata")
            answer_data = tf.placeholder(tf.float32, shape=[None, self.numAnswers, self.embedDim], name="answerdata") #1hot vector of answer

            # Initialize random embeddings
            # Initialize as normal distribution with mean = 0 and std.deviation = 1 according to paper
            # To perform matrix multiplication on higher dimensions
            batchSizing= tf.shape(story_data)[0]


            tempDebug = answer_data

            '''
            # Parameters to train
            Z = tf.Variable(tf.truncated_normal([self.vocabularySize, self.word2VecDim])) # TODO: Initialize as pre-trained word vector
            T = tf.Variable(tf.truncated_normal([self.word2VecDim, self.embedDim]))
            # To encode temporal information on which sentence we are currently on
            memoryA = tf.Variable(tf.truncated_normal([self.maxNumSentences, self.embedDim], stddev=0.05))
            memoryB = tf.Variable(tf.truncated_normal([self.maxNumSentences, self.embedDim], stddev=0.05))
                
            # TODO: only multiply the embeddings for question, answer, and plots
            # (vocabularySize, embedDim)
            embeddings_A = tf.matmul(Z, T)
            embeddings_B = tf.matmul(Z, T)
            embeddings_C = tf.matmul(Z, T)
            embeddings_F = tf.matmul(Z, T)

            # (b, z, v) * (v, d)  = (b, z, d)
            answerG = tf.reshape(tf.matmul(tf.reshape(story_data, (batchSizing*self.maxNumSentences,self.vocabularySize)), embeddings_F), (batchSizing, self.maxNumSentences, self.embedDim))

            memory_matrix_m = tf.reshape(tf.matmul(tf.reshape(story_data, (batchSizing*self.maxNumSentences,self.vocabularySize)), embeddings_A), (batchSizing, self.maxNumSentences, self.embedDim))

            # Hidden layers for word encodings (sum words to get sentence representation)
            # This gets a sentence representation for each sentence in a paragraph
            # Gets a single sentence representation for that 1 question
            control_signal_u= tf.matmul(tf.reshape(question_data, (batchSizing, self.vocabularySize)), embeddings_B)

            # Get training control values
            c_set = tf.reshape(tf.matmul(tf.reshape(story_data, (batchSizing*self.maxNumSentences,self.vocabularySize)), embeddings_C), (batchSizing, self.maxNumSentences, self.embedDim))

            # Add temporal information
            memory_matrix_m = tf.add(memory_matrix_m, memoryA)
            c_set = tf.add(c_set, memoryB)

            # Use memory multplied with control to select a story
            # (b,z,d) * (b,d,1) = (b,z,1)
            memory_selection = tf.reshape(tf.matmul(memory_matrix_m, tf.reshape(control_signal_u, (batchSizing, self.embedDim, 1))), (batchSizing, self.maxNumSentences, 1))
            # Calculate which story to select
            p = tf.nn.softmax(memory_selection, 1)
            # Select the story
            c_set = tf.transpose(c_set, (0, 2, 1))
            o = tf.reshape(tf.matmul(c_set, tf.reshape(p,(batchSizing, self.maxNumSentences, 1))),(batchSizing, self.embedDim))
            # Calculate the sum
            # (b, 1, d)
            o_u_sum = tf.add(o, control_signal_u)
            # answerG => (b, z, d) , z = numberSentence => numberAnswerChoices in this case
            # TODO: Modify to get: (b, d) * (b, d, z) => (b, z) => For every batch, there is an answer
            predicted_answer_labels = tf.matmul(o_u_sum, tf.transpose(answerG, (0, 2, 1)))
            # (b,1,d) * (d, V) = (b, V) => Choose a vocabulary as an answer

            # No idea if the bottom line is still needed
            # predicted_answer_labels = tf.reshape(predicted_answer_labels, [-1, 1, self.vocabularySize]) 
            predicted_answer_labels = tf.reshape(predicted_answer_labels, [-1, 1, self.vocabularySize]) 
            y_predicted = predicted_answer_labels
            answer_data = tf.reshape(answer_data, [batchSizing, 1, self.vocabularySize])
            y_target = answer_data
            # Multi-class Classification
            argyPredict  = tf.argmax(y_predicted,2)
            argyTarget = tf.argmax(y_target,2)
            correctPred = tf.equal(argyPredict, argyTarget)
            accuracy = tf.reduce_mean(tf.cast(correctPred, "float"))
            # Paper said it didn't average the loss, but it will reach infinity if batch size is too large
            loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits = y_predicted, labels = y_target))
            #Optimizer
            optimizer = tf.train.AdagradOptimizer(learningRate).minimize(loss)
            '''

            # TODO: Uncomment once done fixing everything above
            # Run session
            loss_values = []
            with tf.Session(graph=graph) as session:
                #Initialize variables
                tf.global_variables_initializer().run()
                total_loss = 0.0
                # Num steps is the total number of questions
                for currEpoch in xrange(self.numEpoch):
                    # X, q, a = ShuffleBatches(X,q,a) # TODO: 
                    numCorrect = 0.0
                    for step in xrange(self.numQuestions/self.batchSize):
                        train_story = self.X[step*self.batchSize:(step+1)*self.batchSize]
                        train_qu = np.reshape(self.q[step*self.batchSize:(step+1)*self.batchSize], (self.batchSize,1, self.word2VecDim))
                        train_answer = np.reshape(self.a[step*self.batchSize:(step+1)*self.batchSize], (self.batchSize, self.numAnswers,self.word2VecDim))
                        feed_dictS = {story_data: train_story, question_data: train_qu, answer_data: train_answer}

                        session.run([tempDebug], feed_dict = feed_dictS)

                        '''
                        _,l,yhat,y, acc, argyhat, argy, correctPrediction = session.run([optimizer, loss, predicted_answer_labels, answer_data, accuracy, argyPredict, argyTarget, correctPred], feed_dict = feed_dictS)

                        #numCorrect += acc # DOesnt work for batchsize > 1
                        #print "CorrectPrediction", correctPrediction
                        total_loss += l
                        numCorrect += sum(correctPrediction)
                    #Store loss values for the epoch
                    loss_values.append(total_loss)
                    accuracyThisEpoch = numCorrect/float(num_steps)
                    valLoss, valAccuracy = session.run([loss, accuracy], feed_dict = feed_dictV)
                    #testLoss, testAccuracy = session.run([loss, accuracy], feed_dict = feed_dictT)
                    print "valLoss", valLoss
                    print "valAcc", valAccuracy
                    #print "testLoss", testLoss
                    #print "testAcc", testAccuracy
                    print 'EpochNum:', currEpoch
                    print 'LearningRate:', learningRate
                    print 'TotalLossCurrEpoch:', total_loss
                    print 'AccuracyCurrEpoch:', accuracyThisEpoch
                    total_loss = 0.0
                    numCorrect = 0.0
                    #if not currEpoch % 25:
                    if not currEpoch % 15:
                        # LearningRate Annealing
                        learningRate = learningRate/2.0 # 4.1 Annealing. 
                print("Training done!")
                '''
            '''
            #Print loss plot
            pylab.ylabel("Loss")
            pylab.xlabel("Step #")
            loss_value_array = np.array(loss_values)
            pylab.plot(np.arange(0,epoch_size, 1),loss_values)
            pylab.show()  
            '''


def runTfIdf(trainPlots, testQuestions):
    # To output results in a beautified html file
    correctHtml = HtmlOutput("correctfile.html")
    wrongHtml = HtmlOutput("wrongfile.html")

    startTime = datetime.datetime.now()

    # Train the plots with TFIDF score
    tfidf_ = TfIdf(trainPlots)

    endTime = datetime.datetime.now()

    # Run validation test
    numChoices = np.array([0, 0, 0, 0, 0])
    corrChoices = np.array([0, 0, 0, 0, 0])
    numQuestions = len(testQuestions)
    finalAnswers = np.zeros(numQuestions)
    numCorrect = 0
    currQaNum = 0
    for currQA in testQuestions:
        currQaNum += 1
        currTotalScore = -1.0
        choice = -1

        # Sentence vectors for question and the 5 possible answers
        QuestionVec = tfidf_.getSentenceVector(currQA.imdb_key, currQA.question)
        AnsA = tfidf_.getSentenceVector(currQA.imdb_key, currQA.answers[0])
        AnsB = tfidf_.getSentenceVector(currQA.imdb_key, currQA.answers[1])
        AnsC = tfidf_.getSentenceVector(currQA.imdb_key, currQA.answers[2])
        AnsD = tfidf_.getSentenceVector(currQA.imdb_key, currQA.answers[3])
        AnsE = tfidf_.getSentenceVector(currQA.imdb_key, currQA.answers[4])

        chosenPlot = ""
        for currPlot in trainPlots[currQA.imdb_key]:
            PlotVec = tfidf_.getSentenceVector(currQA.imdb_key, currPlot)
            matrix = np.array([PlotVec, QuestionVec, AnsA, AnsB, AnsC, AnsD, AnsE])
            matrixSparse = sparse.csr_matrix(matrix)

            # Calculates scores using cosine similarity
            scores = cosine_similarity(matrixSparse)
            plotQuestionScore = scores[0, 1]
            answerScores = np.array([scores[0,2], scores[0,3],
                           scores[0,4], scores[0,5], scores[0,6]])
            plotAnswerScore = max(answerScores)
            score = (plotQuestionScore + plotAnswerScore)/2.0

            # Find plot with maximum score
            if  score > currTotalScore:
                currTotalScore = score
                choice = np.argwhere(answerScores == max(answerScores))[0,0]
                chosenPlot = currPlot

        # Update correct and chosen answer choices
        numChoices[choice] += 1
        corrChoices[currQA.correct_index] += 1

        # Plot output to html files
        if choice == currQA.correct_index:
            numCorrect += 1
            correctHtml.formTable(currQA, chosenPlot, choice, currTotalScore)
        else:
            wrongHtml.formTable(currQA, chosenPlot, choice, currTotalScore)
        finalAnswers[currQaNum-1] = choice

    logTestPeformance(numQuestions, numChoices, corrChoices, numCorrect)
    logTimeInfo(startTime, endTime, "Training Time")

    correctHtml.close() 
    wrongHtml.close()
    return finalAnswers

def logTestPeformance(numQuestions, numChoices, corrChoices, numCorrect):
    ''' Logs the test accuracy performances '''
    logging.info('Number of Questions: ' + str(numQuestions))
    logging.info('Correct Answer Choices: ' + str(corrChoices))
    logging.info('Chosen Answer Choices: ' + str(numChoices))
    logging.info('Number of Correct Answers: ' + str(numCorrect))
    logging.info('Accurary: ' + str((numCorrect*1.0)/numQuestions))

def logElapsedTime(elapsedTime, message):
    ''' Logs the elapsedTime with a given message '''
    hours, remainder = divmod(elapsedTime.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    totalDays = elapsedTime.days
    logging.info(str(message) + ': Days: ' + str(totalDays) +  " hours: " + str(hours) + ' minutes: ' + str(minutes) +  ' seconds: ' + str(seconds))

def logTimeInfo(startTime, endTime, message):
    ''' Logs information about elapsedTime '''
    elapsedTime = endTime - startTime
    logElapsedTime(elapsedTime, message)
    
if __name__ == "__main__":
    logging.basicConfig(filename='output.log', level=logging.DEBUG)
    startTime = datetime.datetime.now()

    # Download the data for training and testing.
    dL = MovieQA.DataLoader()
    [storyTrain, qaTrain]  = dL.get_story_qa_data('train', 'plot')
    finalAnswers = runMemN2N(storyTrain, qaTrain)

    '''
    dL = MovieQA.DataLoader()
    [storyTrain, qaTrain]  = dL.get_story_qa_data('train', 'plot')

    # Run the TFIDF algoritm
    finalAnswers = runTfIdf(storyTrain, qaTrain)

    endTime = datetime.datetime.now()

    logTimeInfo(startTime, endTime, "Total Time")

    # Write the chosen answers from the algorithm
    # in the format that is used for submission to MovieQA leaderboard.
    file = open('testResults.txt', "w")
    file.write('Answers Chosen')
    count = 0
    for answerPicked in finalAnswers:
        file.write('test:' + str(count) + ' ' + str(int(answerPicked)))
        count += 1
    file.close()
    '''

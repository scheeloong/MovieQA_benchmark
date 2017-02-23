from src.htmloutput import HtmlOutput 
import datetime
import logging
import math

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
    logging.info(str(message) + ': Days: ' + str(totalDays) +  
        " hours: " + str(hours) + ' minutes: ' + str(minutes) +  
        ' seconds: ' + str(seconds))

def log_time_info(func):
    def decorated(self,*kargs, **kwargs):
        start_time = datetime.datetime.now()
        ret = func(self,*kargs, **kwargs)
        end_time = datetime.datetime.now() 
        logElapsedTime(end_time-start_time, func.__name__)
        return ret
    return decorated

logging.basicConfig(filename='fusion_out.log', level=logging.DEBUG)
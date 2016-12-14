
class HtmlOutput(object):
    ''' This class outputs analyze results into a HTML file '''

    def __init__(self, htmlFileName):
        '''
        Initialize the html outputs for the file
        '''
        self.htmlFile = open(htmlFileName, "w")
        self.htmlFile.write("<HEAD>")
    
    def formTable(self, currQA, chosenPlot, choice, currTotalScore):
        ''' Forms the html tables for answers and questions for analysis '''
        self.htmlFile.write("<table>")
        self.htmlFile.write(' <tr><td>')
        self.htmlFile.write(currQA.question)
        self.htmlFile.write(' </td></tr>')
        answerChoice = 0
        for answer in currQA.answers:
            self.htmlFile.write(' <tr><td>')
            self.htmlFile.write(str(answerChoice) + ") " + str(answer))
            self.htmlFile.write(' </td></tr>')
            answerChoice += 1
        self.htmlFile.write(' <tr><td>')
        self.htmlFile.write("Plot: " + str(chosenPlot))
        self.htmlFile.write(' </td></tr>')
        self.htmlFile.write(' <tr><td>')
        self.htmlFile.write("choice: " + str(choice) + " answer: " + str(currQA.correct_index) + " score: " + str(currTotalScore))
        self.htmlFile.write(' </td></tr>')
        self.htmlFile.write('</table>')
        self.htmlFile.write('</br>')

    def close(self):
        ''' Close the file '''
        self.htmlFile.write("</HEAD>")
        self.htmlFile.close()

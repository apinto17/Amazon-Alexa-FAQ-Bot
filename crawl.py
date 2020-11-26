from bs4 import BeautifulSoup
import requests
from nltk.tokenize import word_tokenize 
from gensim.models import Word2Vec 
import re
from nltk.stem import PorterStemmer
import nltk
import numpy as np
from numpy.linalg import norm
from sklearn.feature_extraction.text import TfidfVectorizer
import json

ps = PorterStemmer()
stopwords = []



def main():
    # get stopwords
    global stopwords
    stopwords_file = open("stopwords.txt", "r")
    for word in stopwords_file:
        # get rid of newline at the end of word
        stopwords.append(word[:-1])
    # get questions and answers
    questions_answers = get_question_and_answers()

    # clean questions and map to answers
    qq_map = map_questions_to_clean_questions(questions_answers)
    qa_map = map_questions_to_answers(questions_answers)

    # store as json
    file = open("question_db.json", "w+")
    data = {"qq_map" : qq_map, "qa_map" : qa_map}
    json.dump(data, file)



def map_questions_to_clean_questions(questions_answers):
    res = {}
    for question, _ in questions_answers:
        res[clean(question)] = question
    
    return res


def map_questions_to_answers(questions_answers):
    res = {}
    for question, answer in questions_answers:
        res[clean(question)] = answer 

    return res


def clean(string):
    string = string.lower()
    string = re.sub("[^A-Za-z]", " ", string).strip()
    words = string.split(" ")

    string = " ".join([ps.stem(w) for w in words if w not in stopwords])
    return string.replace("  ", " ")



def get_question_and_answers():
    # make a sites list to add all the sites to be scraped
    sites_list = []

    # get q&a's from cal poly page
    sites_list.append(Visit("https://visit.calpoly.edu/faq/faqs.html"))
    sites_list.append(Student("https://afd.calpoly.edu/student-accounts/guides/faq-student"))
    sites_list.append(OpenHouse("https://www.calpoly.edu/faq-open-house"))
    sites_list.append(IncomingStudent("https://orientation.calpoly.edu/cal-poly-faqs"))
    sites_list.append(Housing("http://www.housing.calpoly.edu/content/frequently-asked-questions"))
    sites_list.append(MathScience("https://csmadvising.calpoly.edu/content/faqs"))
    sites_list.append(DRC("https://drc.calpoly.edu/content/support/faq"))
    # get all questions and answers from sites
    res = []
    for site in sites_list:
        res.extend(site.get_questions_and_answers())

    return res



class Site():

    def __init__(self, url):
        r = requests.get(url)
        soup = BeautifulSoup(r.text, "html.parser")
        self.url = url 
        self.soup = soup


    def get_questions_and_answers(self):
        pass



class Visit(Site):

    def get_questions_and_answers(self):
        # parse out questions and answers
        qa_container = self.soup.select_one("ul.faq")
        qa_list = qa_container.select("li")

        # make a mapping from questions to answers
        questions_answers = []
        for qa in qa_list:
            question = qa.select_one("p.accordianLink")
            answer = qa.select_one("div.answerBlock > p")
            if(question is not None and answer is not None):
                questions_answers.append((question.text, answer.text))
            
        return questions_answers


class Student(Site):

    def get_questions_and_answers(self):
        # parse out questions and answers
        qa_container = self.soup.select_one("ul.accordion")
        qa_list = qa_container.select("li")

        # make a mapping from questions to answers
        questions_answers = []
        for qa in qa_list:
            question = qa.select_one("a.accordion-title")
            answer = qa.select_one("div.accordion-content")
            if(question is not None and answer is not None):
                questions_answers.append((question.text, answer.text))
            
        return questions_answers


class OpenHouse(Site):

    def get_questions_and_answers(self):
        # parse out questions and answers
        qa_container = self.soup.find("article", {"role" : "article"})
        qa_list = qa_container.select("div > div > collapsible")

        # make a mapping from questions to answers
        questions_answers = []
        for qa in qa_list:
            question = qa.find("div", {"slot" : "title"})
            answer = qa.find("div", {"slot" : "content"})
            if(question is not None and answer is not None):
                questions_answers.append((question.text, answer.text))
            
        return questions_answers


class IncomingStudent(Site):

    def get_questions_and_answers(self):
        # parse out questions and answers
        qa_list = self.soup.select("div.accordion")

        # make a mapping from questions to answers
        questions_answers = []
        for qa in qa_list:
            question = qa.select_one("h3")
            answer = qa.select_one("p")
            if(question is not None and answer is not None):
                questions_answers.append((question.text, answer.text))            
        
        return questions_answers


class Housing(Site):

    def get_questions_and_answers(self):
        # parse out questions and answers
        qa_list = self.soup.select("div.accordion")

        # make a mapping from questions to answers
        questions_answers = []
        for qa in qa_list:
            question = qa.select_one("p:nth-child(2)")
            answer = qa.select_one("p:nth-child(1)")
            if(question is not None and answer is not None):
                questions_answers.append((question.text, answer.text))            
        
        return questions_answers


class MathScience(Site):

    def get_questions_and_answers(self):
        # parse out questions and answers
        qa_container = self.soup.select_one("div.field-item")
        qa_list = qa_container.select("div.accordion")

        # make a mapping from questions to answers
        questions_answers = []
        for qa in qa_list:
            question = qa.select_one("div:nth-child(1)")
            answer = qa.select_one("div:nth-child(2)")
            if(question is not None and answer is not None):
                questions_answers.append((question.text, answer.text))            
        
        return questions_answers


class DRC(Site):

    def get_answers(self, qa_container):
        result = []
        answers = qa_container.findAll("p")

        for answer in answers:
            if(len(answer.text) > 5 and not answer.text.startswith("For students living on-campus, please visit the Res")):
                result.append(answer)

        return result


    def get_questions_and_answers(self):
        # parse out questions and answers
        qa_container = self.soup.select_one("div.field-item")

        questions = self.soup.findAll("a", {"id" : re.compile("^q")})
        answers = self.get_answers(qa_container)[:-1]

        total = len(questions)
        # make a mapping from questions to answers
        questions_answers = []
        for i in range(total):
            question = questions[i].next
            answer = answers[i].text
            if(question is not None and answer is not None):
                questions_answers.append((question, answer))            
        
        return questions_answers
        



if(__name__ == "__main__"):
    main()
from bs4 import BeautifulSoup
import requests
from nltk.tokenize import word_tokenize 
from gensim.models import Word2Vec 
import re
from nltk.stem import PorterStemmer
import nltk
from nltk.corpus import stopwords
import numpy as np
from numpy.linalg import norm
from sklearn.feature_extraction.text import TfidfVectorizer
import json

stopwords = stopwords.words('english')
ps = PorterStemmer()



def main():
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

    return " ".join([ps.stem(w) for w in words if w not in stopwords])



def get_question_and_answers():
    # make a sites list to add all the sites to be scraped
    sites_list = []

    # get q&a's from cal poly page
    sites_list.append(Housing("https://visit.calpoly.edu/faq/faqs.html"))

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



class Housing(Site):

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
        



if(__name__ == "__main__"):
    main()
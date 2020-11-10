import sys
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

stopwords = stopwords.words('english')
ps = PorterStemmer()


METHOD = "tfidf"
vocab = []



def main():
    # get questions and answers
    question = sys.argv[1]
    questions_answers = get_question_and_answers()

    # clean questions and map to answers
    qq_map = map_questions_to_clean_questions(questions_answers)
    qa_map = map_questions_to_answers(questions_answers)

    # train model
    train_data = get_training_data(qa_map)
    # get vocab from training data
    global vocab
    vocab = get_vocab(train_data)

    # choose method and appropriate model
    model = None
    if(METHOD == "word2vec"):
        model = Word2Vec(train_data, min_count=1, window=3)
    elif(METHOD == "tfidf"):
        train_data_temp = [" ".join(sent) for sent in train_data]
        model = TfidfVectorizer().fit(train_data_temp)

    # get most similar question
    question = process_question(model, question)
    most_similar_question = get_most_similar_question(model, train_data, question)

    # output question and answer
    print("Similar Question:")
    print(qq_map[most_similar_question])
    print("Answer:")
    print(qa_map[most_similar_question])


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


def get_vocab(train_data):
    res = []
    for sent in train_data:
        for word in sent:
            res.append(word)

    return list(set(res))


def get_most_similar_question(model, train_data, question):
    cosine_dict ={}

    # find most similar question
    a = get_vec(model, question)
    for train_question in train_data:
        b = get_vec(model, train_question)
        a, b = normalize_length(a, b)
        # cosine similarity between question and train_question
        cos_sim = np.vdot(a, b)/(norm(a) * norm(b))

        # join train_question back into single string
        train_question = " ".join(train_question)
        cosine_dict[train_question] = cos_sim

    # put in sorted list of tuples
    dist_sort = sorted(cosine_dict.items(), key=lambda dist: dist[1],reverse = True) ## in Descedning order 

    # return most similar question
    return dist_sort[0][0]


def get_vec(model, data):
    if(METHOD == "tfidf"):
        data = " ".join(data)
        return model.transform([data]).todense()
    elif(METHOD == "word2vec"):
        return model[data]
    else:
        raise ValueError("constant \"METHOD\" incorrectly specified")


def normalize_length(a, b):
    if(len(a) > len(b)):
        a = a[:len(b)]
    elif(len(b) > len(a)):
        b = b[:len(a)]
    
    return a, b


def process_question(model, question):
    res = []
    tokenenized_question = word_tokenize(clean(question))
    for word in tokenenized_question:
        if(word in vocab):
            res.append(word)

    return res



def get_training_data(qa_map):
    # extract questions 
    train_data = []
    questions = qa_map.keys()

    # add questions to train_data
    for question in questions:
        train_data.append(word_tokenize(question))

    return train_data



def clean(string):
    string = string.lower()
    string = re.sub("[^A-Za-z]", " ", string).strip()
    words = string.split(" ")

    return " ".join([ps.stem(w) for w in words if w not in stopwords])



def get_question_and_answers():
    # get q&a's from cal poly page
    r = requests.get("https://visit.calpoly.edu/faq/faqs.html")
    soup = BeautifulSoup(r.text, "html.parser")

    # parse out questions and answers
    qa_container = soup.select_one("ul.faq")
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
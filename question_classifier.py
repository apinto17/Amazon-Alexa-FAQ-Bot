import sys
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

METHOD = "tfidf"
vocab = []



def main():
    # get stopwords
    global stopwords
    stopwords_file = open("stopwords.txt", "r")
    for word in stopwords_file:
        # get rid of newline at the end of word
        stopwords.append(word[:-1])
    # get questions and answers
    question = sys.argv[1]
    file = open("question_db.json", "r")
    data = json.load(file)
    qq_map = data["qq_map"]
    qa_map = data["qa_map"]

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
    print(a)
    for train_question in train_data: 
        # print("\n\n----------------------------------------------")
        # print("Main Question:")
        # print(question)
        # print("\n\nTrain Question:")
        # print(train_question)
        b = get_vec(model, train_question)
        a, b = normalize_length(a, b)
        # cosine similarity between question and train_question
        f = (norm(a) * norm(b))
        if f == 0.0:
            #print(f)
            #print(a)
            #print(b)
            continue
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
        return model.transform([data]).todense().tolist()
    elif(METHOD == "word2vec"):
        return model[data]
    else:
        raise ValueError("constant \"METHOD\" incorrectly specified")


def normalize_length(a, b):
    max_length_0 = max(len(a), len(b))
    max_length_1 = max(len(a[0]), len(b[0]))

    padded_array_a = np.zeros((max_length_0, max_length_1))
    padded_array_b = np.zeros((max_length_0, max_length_1))

    for i in range(len(a)):
        for j in range(len(a[0])):
            padded_array_a[i][j] = a[i][j]

    for i in range(len(b)):
        for j in range(len(b[0])):
            padded_array_b[i][j] = b[i][j]

    return padded_array_a, padded_array_b

    

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



if(__name__ == "__main__"):
    main()
import wikipedia
import re
from bs4 import BeautifulSoup
import requests
import pickle
import nltk
from nltk.corpus import stopwords
import json


stop = stopwords.words('english')


NOUNS = ['NN', 'NNS', 'NNP', 'NNPS']
VERBS = ['VB', 'VBG', 'VBD', 'VBN', 'VBP', 'VBZ']



def clean_document(document):
    """Remove enronious characters. Extra whitespace and stop words"""
    document = re.sub('[^A-Za-z .-]+', ' ', document)
    document = ' '.join(document.split())
    document = ' '.join([i for i in document.split() if i not in stop])
    return document

def tokenize_sentences(document):
    sentences = nltk.sent_tokenize(document)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    return sentences

def get_entities(document):
    """Returns Named Entities using NLTK Chunking"""
    entities = []
    sentences = tokenize_sentences(document)

    # Part of Speech Tagging
    sentences = [nltk.pos_tag(sent) for sent in sentences]
    for tagged_sentence in sentences:
        for chunk in nltk.ne_chunk(tagged_sentence):
            if type(chunk) == nltk.tree.Tree:
                entities.append(' '.join([c[0] for c in chunk]).lower())
    return entities

def word_freq_dist(document):
    """Returns a word count frequency distribution"""
    words = document.split()
    words = [word.lower() for word in words if word not in stop]
    fdist = nltk.FreqDist(words)
    return fdist

def extract_subject(document):
    # Get most frequent Nouns
    fdist = word_freq_dist(document)
    most_freq_nouns = [w for w, c in fdist.most_common(10)
                       if nltk.pos_tag([w])[0][1] in NOUNS]

    # Get Top 10 entities
    entities = get_entities(document)
    top_10_entities = [w for w, c in nltk.FreqDist(entities).most_common(10)]

    # Get the subject noun by looking at the intersection of top 10 entities
    # and most frequent nouns. It takes the first element in the list
    subject_nouns = [entity for entity in top_10_entities
                    if entity.split()[0] in most_freq_nouns]
    if(len(subject_nouns) > 0):
        return subject_nouns[0]
    else:
        return None

def get_subject(document):
    document = clean_document(document)
    subject = extract_subject(document)

    return subject

def get_sent_w_subject(section, subject):
    if(subject is None):
        return None
    sents = nltk.sent_tokenize(section)

    for sent in sents:
        if(subject in sent.lower()):
            return sent


def main():
    # split by section
    raw_content = wikipedia.page("California Polytechnic State University").content
    titles_removed = re.sub("=== .* ===", "*****", raw_content).strip()
    titles_removed = re.sub("== .* ==", "*****", titles_removed)
    sections = titles_removed.split("*****")

    key_words = []
    for i in range(len(sections)):
        # get subject
        print(sections[i])
        document = clean_document(sections[i])
        subject = extract_subject(document)

        # get sentence containing subject
        sent_w_subject = get_sent_w_subject(document, subject)

        if(subject is None):
            continue
        key_words.append((subject, sent_w_subject))

    print(key_words)



if(__name__ == "__main__"):
    main()



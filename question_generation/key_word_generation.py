import wikipedia
import re
import nltk
from nltk.corpus import stopwords

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
    most_freq_nouns = [w for w, c in fdist.most_common(20)
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


def extract_other_targets(document):
    # Get most frequent Nouns
    fdist = word_freq_dist(document)
    most_freq_nouns = [w for w, c in fdist.most_common(20)
                       if nltk.pos_tag([w])[0][1] in NOUNS]
    most_freq_verbs = [w for w, c in fdist.most_common(1000)
                       if nltk.pos_tag([w])[0][1] in VERBS]


    print("most freq noun:", most_freq_nouns)
    print("most freq verbs:", most_freq_verbs)

    # Get Top 10 entities
    entities = get_entities(document)

    top_10_entities = [w for w, c in nltk.FreqDist(entities).most_common(10)]

    print("entities are:", top_10_entities)

    return most_freq_nouns + most_freq_verbs + top_10_entities


    # # Get the subject noun by looking at the intersection of top 10 entities
    # # and most frequent nouns. It takes the first element in the list
    # subject_nouns = [entity for entity in top_10_entities
    #                 if entity.split()[0] in most_freq_nouns]
    # if(len(subject_nouns) > 0):
    #     return subject_nouns[0]
    # else:
    #     return None




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
    return None


def get_wiki_sections():
    # split by section
    raw_content = wikipedia.page("California Polytechnic State University").content
    titles_removed = re.sub("=== .* ===", "*****", raw_content).strip()
    titles_removed = re.sub("== .* ==", "*****", titles_removed)
    sections = titles_removed.split("*****")
    return sections

def Get_tartet_word_and_sentence(sections): #returns list of list of tuples, inner list is by section
    
    key_words = []
    for i in range(1): #len(sections)
        this_section = []
        # get subject
        print(sections[i])
        document = clean_document(sections[i])
        subject = extract_subject(document)

        my_targets_Lst = extract_other_targets(document)

        my_targets_Lst = list(set(my_targets_Lst))
        # get sentence containing subject
        
        if(subject != None):
            sent_w_subject = get_sent_w_subject(sections[i], subject)
            if sent_w_subject != None:
                this_section.append((subject, sent_w_subject))

        print("my targets set:" , set(my_targets_Lst))
        print("my targets list:", my_targets_Lst)
        for word in my_targets_Lst:
            sent_w_subject = get_sent_w_subject(sections[i], word)
            if sent_w_subject != None:
                this_section.append((word, sent_w_subject))



        key_words.append(this_section)

    print("\n\n")
    for w,_ in key_words[0]:
        print(w)
    print(len(key_words[0]))



def main():
    # split by section
    raw_content = wikipedia.page("California Polytechnic State University").content
    titles_removed = re.sub("=== .* ===", "*****", raw_content).strip()
    titles_removed = re.sub("== .* ==", "*****", titles_removed)
    sections = titles_removed.split("*****")

    key_words = []
    for i in range(1): #len(sections)
        this_section = []
        # get subject
        print(sections[i])
        document = clean_document(sections[i])
        subject = extract_subject(document)

        my_targets_Lst = extract_other_targets(document)

        # get sentence containing subject
        
        if(subject != None):
            sent_w_subject = get_sent_w_subject(sections[i], subject)
            if sent_w_subject != None:
                this_section.append((subject, sent_w_subject))

        print("my targets list:", my_targets_Lst)
        for word in my_targets_Lst:
            sent_w_subject = get_sent_w_subject(sections[i], word)
            if sent_w_subject != None:
                this_section.append((word, sent_w_subject))



        key_words.append(this_section)

    print("\n\n")
    for w,_ in key_words[0]:
        print(w)
    print(len(key_words[0]))



if(__name__ == "__main__"):
    #main()
    wiki_sections = get_wiki_sections()
    key_words_list = Get_tartet_word_and_sentence(wiki_sections)


from nltk.stem.porter import PorterStemmer
from copy import deepcopy
from collections import defaultdict
import math
import gensim
from gensim.parsing.preprocessing import remove_stopwords, preprocess_string


class TextProcessor:
    def __init__(self, text):
        self.stopword_list_tfidf = []
        self.raw_text = text
        self.text_as_dict = None
        self.classes = []
        self.texts = []
        self.vocabulary = set()

    def convert_text_to_dictionary(self):
        texts = self.raw_text.split('\n')
        d = defaultdict(list)
        for text in texts:
            splitted_text = text.split('\t')
            if len(splitted_text) == 2:
                self.texts.append(self.create_text_list_with_seperate_words(splitted_text[1]))
                d[splitted_text[0]].append(self.create_text_list_with_seperate_words(splitted_text[1]))
                self.classes.append(splitted_text[0])
        self.text_as_dict = d

    def create_text_list_with_seperate_words(self, text):
        l = []
        words = text.split(" ")
        for word in words:
            l.append(word)
            self.vocabulary.add(word)
        return l

    def remove_stopwords_tfidf(self):
        l = []
        for i in range(len(self.texts)):
            l.append([])
            for word in self.texts[i]:
                if word in self.stopword_list_tfidf:
                    continue
                else:
                    l[i].append(word)
        self.texts = deepcopy(l)

        l1 = []
        for word in self.vocabulary:
            if word in self.stopword_list_tfidf:
                continue
            else:
                l1.append(word)
        self.vocabulary = deepcopy(l1)

        if self.text_as_dict is not None:
            for k, v in self.text_as_dict.items():
                l2 = []
                for i in range(len(list(self.text_as_dict[k]))):
                    l2.append([])
                    for word in self.text_as_dict[k][i]:
                        if word in self.stopword_list_tfidf:
                            continue
                        else:
                            l2[i].append(word)
                self.text_as_dict[k] = deepcopy(l2)

    def create_tfidf_stopword_list(self, threshold, weight=0.5):
        t = threshold
        treshold_dict = defaultdict(float)
        # Create Treshold Dict -> word : threshold
        for word in self.vocabulary:
            treshold_dict[word] = 0.0
        # Get IDF: log(|N|/nt)
        # first get |N| and nt per word
        nt = defaultdict(int)
        N = 0
        for k, v in self.text_as_dict.items():
            for text in v:
                N += 1
                done_words = []
                for word in text:
                    if word not in done_words:
                        nt[word] += 1
                    done_words.append(word)
        # calc IDF per word
        for k, v in nt.items():
            x = nt[k]
            nt[k] = math.log2(N) - math.log2(x)

        # Get TF per word per document: count(word) / count(most frequent word)
        tfidf_w = defaultdict(list)
        for k, v in self.text_as_dict.items():
            for text in v:
                d = defaultdict(int)
                for word in text:
                    d[word] += 1
                c_max = max(d.values())
                for word, count in d.items():
                    x = d[word]
                    d[word] = weight + (weight * x / c_max)
                    tfidf = d[word] * nt[word]
                    tfidf_w[word].append(tfidf)

        # mean the tfidf scores per word
        for word, tfidfs in tfidf_w.items():
            n = len(tfidfs)
            s = 0
            for i in tfidfs:
                s += i
            tfidf_w[word] = s / n

        # create stop word list
        sl = []
        for k, v in tfidf_w.items():
            if v <= threshold:
                sl.append(k)
        self.tfidf_scores = tfidf_w
        self.stopword_list_tfidf = sl

    def stemm_text(self):
        stemmer = PorterStemmer()
        l = []
        for i in range(len(self.texts)):
            l.append([])
            for word in self.texts[i]:
                word = word.lower()
                l[i].append(stemmer.stem(word))
        self.texts = deepcopy(l)
        if self.text_as_dict is not None:
            for k, v in self.text_as_dict.items():
                l2 = []
                for i in range(len(list(self.text_as_dict[k]))):
                    l2.append([])
                    for word in self.text_as_dict[k][i]:
                        l2[i].append(stemmer.stem(word))
                self.text_as_dict[k] = deepcopy(l2)


def get_data():
    # Load Data
    train_str = ""
    with open("C:\\Users\\weinm\\Desktop\\train.txt", encoding='utf8') as fhi:
        train = fhi.readlines()
    for l in train:
        train_str += l

    test_str = ""
    with open("C:\\Users\\weinm\\Desktop\\train.txt", encoding='utf8') as fho:
        test = fho.readlines()
    for l in test:
        test_str += l
    # Preprocess Train Data
    gensim.parsing.preprocessing.strip_punctuation(train_str)
    train_processor = TextProcessor(train_str)
    train_processor.convert_text_to_dictionary()
    train_processor.stemm_text()
    # Why is the threshold so big though??
    train_processor.create_tfidf_stopword_list(3.5)
    train_processor.remove_stopwords_tfidf()


    #  Preprocess Test Data
    gensim.parsing.preprocessing.strip_punctuation(test_str)
    test_processor = TextProcessor(test_str)
    test_processor.convert_text_to_dictionary()
    test_processor.stemm_text()
    test_processor.remove_stopwords_tfidf()

    y_train = train_processor.classes
    X_train = train_processor.texts

    y_test = test_processor.classes
    X_test = test_processor.texts

    return y_train, X_train, y_test, X_test, test_processor, train_processor
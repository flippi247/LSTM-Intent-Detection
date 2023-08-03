from Preprocessing import get_data
from collections import defaultdict
from copy import deepcopy
import fasttext
from Tsne import tsne_plot



# Get Train/Testdata
y_train, X_train, y_test, X_test, test_processor, train_processor = get_data()


# Fasttext Models with Labels in Front
with open("C:\\Users\\weinm\\Desktop\\train_new.txt", 'w', encoding='utf-8')as fho:
    for i in range(len(y_train)):
        sent = ""
        for word in X_train[i]:
            sent += word
            if X_train[i].index(word) != len(X_train[i])-1:
                sent += ' '
        # y_Label only needed for supervised tasks
        y_label_fasttext_format = '__label__' + y_train[i] + ' '
        fho.write(y_label_fasttext_format + sent+'\n')

with open("C:\\Users\\weinm\\Desktop\\test_new.txt", 'w', encoding='utf-8')as fho:
    for i in range(len(y_test)):
        sent = ""
        for word in X_test[i]:
            sent += word
            if X_test[i].index(word) != len(X_test[i])-1:
                sent += ' '
            # y_Label only needed for supervised tasks
            y_label_fasttext_format = '__label__' + y_test[i] + ' '
            fho.write(y_label_fasttext_format + sent+'\n')

# Model :
model = fasttext.train_supervised(input='C:\\Users\\weinm\\Desktop\\train_new.txt', dim=200, minCount=3, epoch=1000)
tokens = [model.get_word_vector(x) for x in model.get_words()]
labels = model.get_words()
tsne_plot(model, labels, tokens)

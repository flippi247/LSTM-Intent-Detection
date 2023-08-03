import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.colors as mcolors

def tsne_plot(model, labels, tokens):
    "Creates and TSNE model and plots it"

    tsne_model = TSNE(n_components=2, random_state=0, perplexity=20, learning_rate=700, n_iter=10000)
    new_values = tsne_model.fit_transform(tokens)[:,:3]

    labels = [model.predict(word)[0][0] for word in model.get_words()]
    color_dict = {}

    i = 0
    ii = 0
    mlabels = model.get_labels()
    for color in mcolors.CSS4_COLORS.keys():
        if ii not in range(len(mlabels)):
            break
        if i % 7 == 0:
            color_dict[mlabels[ii]] = color
            ii += 1
        i += 1

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    words = model.get_words()
    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i], y[i], color=color_dict[labels[i]])
        plt.annotate(words[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()


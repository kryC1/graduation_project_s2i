import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.signal import argrelextrema
import math


def rev_sigmoid(x: float) -> float:
    return (1 / (1 + math.exp(0.5 * x)))


def activate_similarities(similarities: np.array, p_size=10) -> np.array:
    x = np.linspace(-10, 10, p_size)
    y = np.vectorize(rev_sigmoid)
    activation_weights = np.pad(y(x), (0, similarities.shape[0] - p_size))
    diagonals = [similarities.diagonal(each) for each in range(0, similarities.shape[0])]
    diagonals = [np.pad(each, (0, similarities.shape[0] - len(each))) for each in diagonals]
    diagonals = np.stack(diagonals)
    diagonals = diagonals * activation_weights.reshape(-1, 1)
    activated_similarities = np.sum(diagonals, axis=0)
    return activated_similarities

def split_paragraph(raw_text, order_input):
    model = SentenceTransformer('all-mpnet-base-v2')
    sentences = raw_text.split('. ')
    embeddings = model.encode(sentences)
    similarities = cosine_similarity(embeddings)
    order_user = 0

    try:
        activated_similarities = activate_similarities(similarities, p_size=4)
    except:
        print("exception negative value, raw_text is returned")
        return raw_text

    if order_input == "less_frequent":
        print("less frequent\n")
        order_user = 3
    elif order_input == "normal":
        print("normal frequent\n")
        order_user = 2
    elif order_input == "more_frequent":
        print("more frequent\n")
        order_user = 1
    else:
        print("i am in the else (less frequent)\n")
        order_user = 1

    minmimas = argrelextrema(activated_similarities, np.less, order=order_user)

    sentece_length = [len(each) for each in sentences]
    long = np.mean(sentece_length) + np.std(sentece_length) * 2
    short = np.mean(sentece_length) - np.std(sentece_length) * 2
    text = ''
    for each in sentences:
        if len(each) > long:
            comma_splitted = each.replace(',', '.')
        else:
            text += f'{each}. '
    sentences = text.split('. ')
    text = ''
    for each in sentences:
        if len(each) < short:
            text += f'{each} '
        else:
            text += f'{each}. '

    split_points = [each for each in minmimas[0]]

    text_to_split = ''
    for num, each in enumerate(sentences):
        if num in split_points:
            text_to_split += f'\n{each}. '
        else:
            text_to_split += f'{each}. '

    return text_to_split
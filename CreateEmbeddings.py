import json
import numpy as np
from textblob import TextBlob

embeddings_index = {}
f = open('glove1/glove.twitter.27B.50d.txt', encoding="utf-8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

f = open('glove/glove.6B.50d.txt', encoding="utf-8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

with open('Data/word_index.json') as json_file:
    vocabs = json.load(json_file)

not_found_words = {}

embedding_matrix = np.zeros((len(vocabs) + 1, 50))
for word, i in vocabs.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is None:
        doc = TextBlob(word)
        word = str(doc.correct())
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is None:
            if word not in not_found_words:
                not_found_words[word] = i
        else:
            embedding_matrix[i] = embedding_vector
    else:
        embedding_matrix[i] = embedding_vector

with open("Data/not_found_words.json", "w") as outfile:
    json.dump(not_found_words, outfile)
    
np.savetxt("Data/embedding_matrix.txt", CreateEmbeddings.embedding_matrix)

#for loading copy this
# content = np.loadtxt('Data/embedding_matrix.txt')

print("embeddings: ",len(embeddings_index), "vocabs", len(vocabs), "not found words", len(not_found_words))
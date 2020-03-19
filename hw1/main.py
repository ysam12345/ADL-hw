import numpy as np

import tensorflow as tf
from tensorflow.keras import layers

import json
from gensim.models.word2vec import Word2Vec


#from nltk.tokenize import word_tokenize


TEST_PATH = './dataset/test.jsonl'
TRAIN_PATH = './dataset/train.jsonl'
VALID_PATH = './dataset/valid.jsonl'

dataset_path = {
    'test': TEST_PATH,
    'train': TRAIN_PATH, 
    'valid': VALID_PATH, 
}

dataset = {}
for dataset_type in dataset_path:
    dataset[dataset_type] = []
    with open(dataset_path[dataset_type], 'r') as t_f:
        for line in t_f:
            dataset[dataset_type].append(json.loads(line))

def extract_sentences(data):
    text = data['text']
    sent_bounds = data['sent_bounds']
    sentenses = []
    for bound in sent_bounds:
        sentenses.append(text[bound[0]:bound[1]] \
            .replace(',','') \
            .replace('.','') \
            .replace('?','') \
            .replace('\n',''))
    return sentenses

#word_tokenize(sentence)
def extract_tokens(sentence):
    return sentence.split(' ')

'''
def get_vectors(tokens):
    model = Word2Vec([tokens],min_count=1,size=100)
    vectors = [model.wv[token] for token in tokens]
    return vectors
'''
print(len(dataset['train']))
count = 0
for data in dataset['train']:
    count +=1
    print(count)
    sentences = extract_sentences(data)
    for i in range(len(sentences)):
        sentences[i] = extract_tokens(sentences[i])
    model = Word2Vec(sentences,min_count=1,size=100)
    vectors = []
    for i in range(len(sentences)):
        vectors.append([])
        for j in range(len(sentences[i])):
            vectors[i].append(model.wv[sentences[i][j]])

model = tf.keras.Sequential()
model.add(layers.Embedding(input_dim=500, output_dim=64))
model.add(layers.LSTM(128))
model.add(layers.Dense(2))

model.summary()
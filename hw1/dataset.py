import json
import numpy as np
from gensim.models.word2vec import Word2Vec

from tqdm import tqdm


TEST_PATH = './dataset/test.jsonl'
TRAIN_PATH = './dataset/train.jsonl'
VALID_PATH = './dataset/valid.jsonl'

dataset_path = {
    'test': TEST_PATH,
    'train': TRAIN_PATH, 
    'valid': VALID_PATH, 
}


def load_dataset_from_json():
    dataset = {}
    for dataset_type in dataset_path:
        dataset[dataset_type] = []
        with open(dataset_path[dataset_type], 'r') as t_f:
            for line in t_f:
                dataset[dataset_type].append(json.loads(line))
    return dataset

def extract_sentences(data):
    text = data['text']
    sent_bounds = data['sent_bounds']
    sentenses = []
    for bound in sent_bounds:
        sentenses.append(text[bound[0]:bound[1]])
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

def prepare_dataset(dataset, data_nums=None):
    X_dataset = []
    Y_dataset = []
    if data_nums is not None:
        progress = tqdm(total=data_nums)
    else:
        progress = tqdm(total=len(dataset))
    count = 0
    for data in dataset:
        progress.update(1)
        count += 1
        if data_nums is not None and count > data_nums:
            break
        extractive_index = data['extractive_summary']
        sentences = extract_sentences(data)
        for i in range(len(sentences)):
            sentences[i] = extract_tokens(sentences[i])
            sentences[i][-1] = sentences[i][-1].replace('\n', '')
        model = Word2Vec(sentences,min_count=1,size=100)
        '''
        embeddings = []
        for i in range(len(sentences)):
            embeddings.append([])
            for j in range(len(sentences[i])):
                embeddings[i].append(model.wv[sentences[i][j]])
        '''
        for i in range(len(sentences)):
            if i == extractive_index:
                label = 1
            else:
                label = 0
            for j in range(len(sentences[i])):
                X_dataset.append(model.wv[sentences[i][j]])
                Y_dataset.append(label)
    progress.close()
    return np.asarray(X_dataset), np.asarray(Y_dataset)
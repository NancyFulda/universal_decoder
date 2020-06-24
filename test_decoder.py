import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import fasttext
from dataset_fasttext import Dataset
import sys

from decoder_model import Universal_Decoder

import random
import time
import json
import pickle as pkl

# ===============================================
#
# Globals

LOAD_FILE = "ouput/40_decoder.pkl"

CORPUS='wikipedia' # text corpus to use when building InferSent vocab

USE_CUDA = True

#INPUT_EMBEDDING = 'FASTTEXT_BOW'
INPUT_EMBEDDING = 'INFERSENT'

if INPUT_EMBEDDING == 'FASTTEXT_BOW':
    EMBEDDING_SIZE = 300
elif INPUT_EMBEDDING == 'INFERSENT':
    EMBEDDING_SIZE = 4096
else:
    raise ValueError('Unkown embedding method: ' + str(INPUT_EMBEDDING))

#
# ===============================================
#
# Corpus Definitions
if CORPUS == 'one_sentence':
    input_filename = 'text_corpora/onesentence.txt'
if CORPUS == 'tiny_shakespeare':
    input_filename = 'text_corpora/tiny_shakespeare.txt'
if CORPUS == 'wikipedia':
    input_filename = 'text_corpora/Wikipedia_first_10000_lines_clean.txt'
if CORPUS == 'wikipedia_large':
    input_filename = 'text_corpora/Wikipedia_text_with_periods_clean.txt'

reconstruction_file = open(input_filename,'r')



#
# ===============================================
#
# Network Parameters

# length of sentences accepted as input
MAX_LEN = 50   #chars    originally set at 512, I'm going to try 30-50
MIN_LEN = 1    #chars

LEARNING_RATE = .00009 #.0001 nancy original value, mayber try .00009 to see if I get slightly more accurate results. 

#FASTTEXT_SIZE = 50000
FASTTEXT_SIZE = 150000
#FASTTEXT_SIZE = 1500000

# Sizes
VOCAB_SIZE = FASTTEXT_SIZE+3
Z_DIMENSION = EMBEDDING_SIZE
DECODER_HIDDEN_SIZE = 600   #originally set to 300

MAX_DECODER_LENGTH = 300 #originally set to 300
NUM_LAYERS_FOR_RNNS = 1 #originally set to 1
CONTEXT_LENGTH = 1 #originally set to 1

#
# ===============================================
#
# Set up embedding models

if INPUT_EMBEDDING in ['FASTTEXT_BOW']:
    with open('data/fasttext.en.pkl','rb') as f:
        data = pkl.load(f)
        fasttext_tokens = data['tokens'][:50000]
        fasttext_vectors = data['vectors'][:50000]

if INPUT_EMBEDDING in ['INFERSENT']:
    import torch
    from infersent.models import InferSent
    #from InferSent import models
    V = 2
    MODEL_PATH = 'infersent/encoder/infersent%s.pkl' % V
    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
    infersent = InferSent(params_model)
    #infersent = models.InferSent(params_model)
    infersent.load_state_dict(torch.load(MODEL_PATH))

    W2V_PATH = 'infersent/fastText/crawl-300d-2M.vec'
    infersent.set_w2v_path(W2V_PATH)

    # get core sentences

    X=[]
    print('Building InferSent vocabulary...')
    with open(input_filename, 'r') as f:
        for line in f:
            sentence = line.strip('\n').strip()
            if len(sentence) > 1:
                X.append(sentence)
    infersent.build_vocab(X, tokenize=True)
    print('Done building vocabulary')

#
# ===============================================
#
# Encoding algorithms

def preprocess(text):
    text = text.replace('\r','').replace('\n','').strip().lower()
    text = text.replace('.', ' .')\
               .replace(',', ' ,')\
               .replace('?', ' ?')\
               .replace('!', ' !')\
               .replace('"', ' "')\
               .replace("'", " '")\
               .replace('-', ' -')\
               .replace(':', ' :')\
               .replace(';', ' ;')
    text = text.replace('  ',' ').replace('  ', ' ').replace('  ', ' ')
    return text

def fasttext_encode(text):
    vector = np.zeros(300)
    text = preprocess(text)
    count = 1
    for word in text.split():
        try:
            vector += fasttext_vectors[fasttext_tokens.index(word)]
            count += 1.0
        except:
            pass

    if count > 0:
        vector = vector/count
    if np.sum(vector) == 0:
        vector = vector+1e5

    return vector

def infersent_encode(text):
    text = preprocess(text)
    if isinstance(text, str):
        #infersent.update_vocab([text])
        embeddings = infersent.encode([text], tokenize=True)
        return embeddings[0]
    else:
        #infersent.update_vocab(text)
        embeddings = infersent.encode(text, tokenize=True)
        return embeddings

#
# ===============================================
#
# Test data

solo_sentences = ["this is a test", "we are friends", "a wise king", "i see a mountain"] 

sentence_pairs = [["we are friends", "you betrayed me"],
                  ["a young boy", "a wise king"],
                  ["i see a mountain", "beauty"]]

print('loading fasttext')
ftext = fasttext.FastText(FASTTEXT_SIZE)
dataset = Dataset(FASTTEXT_SIZE,max_len=MAX_LEN, min_len=MIN_LEN)

print("Loading model...")
decoder = Universal_Decoder(dataset,
                            VOCAB_SIZE,
                            ftext,
                            Z_DIMENSION,
                            DECODER_HIDDEN_SIZE,
                            MAX_DECODER_LENGTH,
                            NUM_LAYERS_FOR_RNNS,
                            USE_CUDA)


def encode_sentence(sentence):
    if INPUT_EMBEDDING == 'FASTTEXT_BOW':
        x = torch.Tensor([fasttext_encode(line)])
    elif INPUT_EMBEDDING == 'INFERSENT':
        x = torch.Tensor([infersent_encode(line)])
    else:
        raise ValueError('Unknown embedding method: ' + str(INPUT_EMBEDDING))

    tokens = ['SOS'] + line.split() + ['EOS']
    y = ftext.get_indices(tokens)
    y = dataset.to_onehot(y, long_type=False)

    if USE_CUDA:
        x = Variable(x.cuda())
        y = y.cuda()
    else:
        x = Variable(x)

    return x,y

def test_sentence(sentence):
    print(sentence)
    x,y = encode_sentence(sentence)
    print(x)
    decoder_outputs, decoder_idxs, rnn_outputs = decoder(x, y, sample=True)
    print('\n')
    print(' '.join(ftext.get_words_from_indices(list(map(int,decoder_idxs)))))


def test_sentence_pair(s1, s2):
    pass

if __name__ == "__main__":
    print('\n')
    for s in solo_sentences:
        test_sentence(s)

    print('\n')
    #for pair in sentence_pairs:
    #    test_sentence_pair(pair[0],pair[1])


    print('\n')
    while(1):
        print("Enter a sentence to test\n")
        sentence = input()
        test_sentence(sentence)

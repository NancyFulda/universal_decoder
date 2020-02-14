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

#
# ===============================================
#
# Globals

SAVE_DIR = 'output/'     # directory to save outputs
LOAD_FILE = None         # path to saved training data

CORPUS='tiny_shakespeare' # text corpus to use during training

USE_CUDA = True
NUM_EPOCHS = 10
SAMPLE_FREQ = 1000
SAVE_FREQ = 100000

INPUT_EMBEDDING = 'FASTTEXT_BOW'
#INPUT_EMBEDDING = 'INFERSENT'

print('Input embedding is ' + INPUT_EMBEDDING)

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
MAX_LEN = 50   #chars
MIN_LEN = 1    #chars

LEARNING_RATE = .0001

FASTTEXT_SIZE = 50000
#FASTTEXT_SIZE = 150000

# Sizes
VOCAB_SIZE = FASTTEXT_SIZE+3
Z_DIMENSION = EMBEDDING_SIZE
DECODER_HIDDEN_SIZE = 300

#MAX_DECODER_LENGTH = 300
MAX_DECODER_LENGTH = 50
NUM_LAYERS_FOR_RNNS = 1
CONTEXT_LENGTH = 1


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
    V = 2
    MODEL_PATH = 'infersent/encoder/infersent%s.pkl' % V
    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
    infersent = InferSent(params_model)
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
# Helper Functions

def output(message):
    print(message)
    with open(SAVE_DIR + 'nohup.out', 'a') as f:
        f.write(message + '\n')


def clean(line):
    line = line.replace('\r','').replace('\n','').strip().replace('  ',' ').replace('  ',' ').lower()
    return line

def get_next_line(source_file):
    line = ''
    while line == '' or line == ' ' or len(line)<MIN_LEN or len(line)>MAX_LEN:
        line = source_file.readline()
        if not line:
            return None
        line = clean(line)
    return line


def track_loss(output_str, output_list):
    output_list=tuple(output_list)
    if total_training_steps % SAMPLE_FREQ == 0:
        print(output_str%output_list)
        print(str(time.time() - start_time) + ' seconds, ' + str((time.time() - start_time)/3600.0) + ' hours')
        print("---------------------------\n")
        with open(SAVE_DIR + 'nohup.out', 'a') as f:
            f.write(output_str%output_list+'\n')
            f.write("---------------------------\n\n")


def show_text(line,y,decoder_idxs):
    # ----------------------------------------------------------------
    # prepare offer
    #if USE_CUDA:
    #    offer = np.argmax(x.cpu().data.numpy(), axis=1).astype(int)
    #else:
    #    offer = np.argmax(x.data.numpy(), axis=1).astype(int)

    # prepare answer
    if USE_CUDA:
        answer = np.argmax(y.cpu().data.numpy(), axis=1).astype(int)
    else:
        answer = np.argmax(y.data.numpy(), axis=1).astype(int)

    # prepare rnn
    rnn_response = list(map(int, decoder_idxs))
        
    # print output
    print("---------------------------")
    #print("Input: ", ' '.join(ftext.get_words_from_indices(offer)))
    print("Input: ", line.strip('\n'))
    print("Target:", ' '.join(ftext.get_words_from_indices(answer)))
    try:
        print("RNN:", ' '.join(ftext.get_words_from_indices(rnn_response)))
    except:
        print("RNN:", '[ERROR] non-ascii character encountered')
    print()
		
    with open(SAVE_DIR + 'nohup.out', 'a') as f:
        f.write("---------------------------\n")
        #f.write("Offer: "+' '.join(ftext.get_words_from_indices(offer))+'\n')
        f.write("Input: " + line)
        f.write("Answer:"+' '.join(ftext.get_words_from_indices(answer))+'\n')
        try:
            f.write("RNN:"+' '.join(ftext.get_words_from_indices(rnn_response))+'\n')
        except:
            f.write("RNN:"+'[ERROR] non-ascii character encountered'+'\n')


def save_data(output_list, decoder, optimizer):
    #save the data
    print('Saving...')

    version = total_training_steps//(10*SAMPLE_FREQ)

    with open(SAVE_DIR + '_losses.txt','a') as f:
        f.write(str(output_list) + '\n')

    print('saving decoder')
    with open(SAVE_DIR + str(version) + '_decoder.pkl', 'wb') as f:
        torch.save(decoder, f)
    with open(SAVE_DIR + str(version) + '_decoder_state_dict.pkl', 'wb') as f:
        torch.save(decoder.state_dict(), f)

    print('saving optimizer')
    with open(SAVE_DIR + str(version) + '_optimizer_state_dict.pkl', 'wb') as f:
        torch.save(optimizer.state_dict(), f)

    #print('saving params')
    #params = {}
    #params['vocab_size']=VOCAB_SIZE
    #params['encoder_hidden_size']=ENCODER_HIDDEN_SIZE
    #params['outer_lstm_hidden_size']=OUTER_LSTM_HIDDEN_SIZE
    #params['z_dimension']=Z_DIMENSION
    #params['num_layers_for_rnns']=NUM_LAYERS_FOR_RNNS
    #params['use_cuda']=USE_CUDA
    #params['decoder_hidden_size']=DECODER_HIDDEN_SIZE
    #params['max_decoder_length']=MAX_DECODER_LENGTH
    #params['max_len']=MAX_LEN
    #params['min_len']=MIN_LEN
    #params['context_length']=CONTEXT_LENGTH
    #params['w2v_window_size']=W2V_WINDOW_SIZE
    #params['sample_freq']=SAMPLE_FREQ
    #params['learning_rate']=LEARNING_RATE
    #params['loss_functions'] = original_loss_functions
    #params['coefficients'] = coefficients
    #with open(SAVE_DIR + '_params.pkl', 'wb') as f:
    #    torch.save(params, f)

    print('done saving')



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
# Training loop

num_epochs = 100
test_frequency = 1

print('loading fasttext')
ftext = fasttext.FastText(FASTTEXT_SIZE)

# init dataset
dataset = Dataset(FASTTEXT_SIZE,max_len=MAX_LEN, min_len=MIN_LEN)

print('initializing models')
if LOAD_FILE is None:
    decoder = Universal_Decoder(dataset,
                            VOCAB_SIZE,
                            ftext,
                            Z_DIMENSION,
                            DECODER_HIDDEN_SIZE,
                            MAX_DECODER_LENGTH,
                            NUM_LAYERS_FOR_RNNS,
                            USE_CUDA)
else:
    print('loading encoder ' + LOAD_FILE + '_decoder...')
    with open(LOAD_FILE + '_decoder.pkl', 'rb') as f:
        decoder = torch.load(f)

optimizer = torch.optim.Adam(decoder.parameters(), lr = LEARNING_RATE)

 
#save me from myself
if SAVE_DIR[-1] != '/':
    SAVE_DIR = SAVE_DIR + '/'

output('beginning calculations')

total_training_steps = 0
start_time = time.time()

for epoch in range(NUM_EPOCHS):
    print("Start epoch!")
    epoch_loss = 0.
    step = 0

    # get the first line in the file
    reconstruction_file.seek(0)
    line = get_next_line(reconstruction_file)

    while(line):
        total_training_steps += 1
        step += 1

        sample = (step % SAMPLE_FREQ == 0)

        output_str = "Epoch: %d, step: %d: "
        output_list = [epoch, step]
        
        #HACK for overfitting
        #line = 'this is a test'

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

        decoder_outputs, decoder_idxs, rnn_outputs = decoder(x, y, sample)

        if sample:
            show_text(line,y,decoder_idxs)

        # mean-squared error
        reconstruction_loss = torch.mean((y - decoder_outputs)*(y-decoder_outputs))
   
        loss = 0
        alpha = 1000 #adjust loss to improve learning speed
        loss += alpha*reconstruction_loss
        output_str += "  loss: %.8f"
        output_list.append(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        track_loss(output_str, output_list)

        if total_training_steps % (SAVE_FREQ) == 0:
            save_data(output_list, decoder, optimizer)
    
        line = get_next_line(reconstruction_file)

    print("\n\nTrained epoch: {}, epoch loss: {}\n\n".format(epoch, epoch_loss))

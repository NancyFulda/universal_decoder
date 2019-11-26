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

#
# ===============================================
#
# Globals

SAVE_DIR = 'output/'     # directory to save outputs
LOAD_FILE = None         # path to saved training data

CORPUS='tiny_shakespeare' # text corpus to use during training

USE_CUDA = True
NUM_EPOCHS = 10
SAMPLE_FREQ = 100

#
# ===============================================
#
# Corpus Definitions

if CORPUS == 'tiny_shakespeare':
    reconstruction_file=open('text_corpora/tiny_shakespeare.txt','r')
if CORPUS == 'wikipedia':
    reconstruction_file=open('data/context_prediction/Wikipedia_text_with_periods_clean.txt','r')


#
# ===============================================
#
# Network Parameters

# length of sentences accepted as input
MAX_LEN = 512   #chars
MIN_LEN = 1    #chars

LEARNING_RATE = .0001

FASTTEXT_SIZE = 50000
#FASTTEXT_SIZE = 150000

# Sizes
VOCAB_SIZE = FASTTEXT_SIZE+3
EMBEDDING_SIZE = 300
Z_DIMENSION = EMBEDDING_SIZE
DECODER_HIDDEN_SIZE = 300

MAX_DECODER_LENGTH = 300
NUM_LAYERS_FOR_RNNS = 1
CONTEXT_LENGTH = 1


#
# ===============================================
#
# Helper Functions

def output(message):
    print(message)
    with open(SAVE_DIR + 'nohup.out', 'a') as f:
        f.write(message + '\n')


def get_next_line(source_file):
    line = source_file.readline()
    while len(line)<MIN_LEN or len(line)>MAX_LEN:
        if not line:
            #source_file.seek(0)
            return None
        line = source_file.readline()


def track_loss(output_str, output_list):
    output_list=tuple(output_list)
    if total_training_steps % SAMPLE_FREQ == 0:
        print(output_str%output_list)
        print(str(time.time() - start_time) + ' seconds, ' + str((time.time() - start_time)/3600.0) + ' hours')
        print("---------------------------\n")
        with open(SAVE_DIR + 'nohup.out', 'a') as f:
            f.write(output_str%output_list+'\n')
            f.write("---------------------------\n\n")


def show_text():
    # ----------------------------------------------------------------
    # prepare offer
    if USE_CUDA:
        offer = np.argmax(x.cpu().data.numpy(), axis=1).astype(int)
    else:
        offer = np.argmax(x.data.numpy(), axis=1).astype(int)

    # prepare answer
    if USE_CUDA:
        answer = np.argmax(y.cpu().data.numpy(), axis=1).astype(int)
    else:
        answer = np.argmax(y.data.numpy(), axis=1).astype(int)

    # prepare rnn
    rnn_response = list(map(int, decoder_idxs))
        
    # print output
    print("---------------------------")
    print("Offer: ", ' '.join(ftext.get_words_from_indices(offer)))
    print("Answer:", ' '.join(ftext.get_words_from_indices(answer)))
    try:
        print("RNN:", ' '.join(ftext.get_words_from_indices(rnn_response)))
    except:
        print("RNN:", '[ERROR] non-ascii character encountered')
		
    with open(SAVE_DIR + 'nohup.out', 'a') as f:
        f.write("---------------------------\n")
        f.write("Offer: "+' '.join(ftext.get_words_from_indices(offer))+'\n')
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

def fasttext_encode(text, dataset):
    vector = np.zeros(300)
    text = dataset.clean(text)
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
    line = 'On your mark, get set, go!'

    while(line):
        line = get_next_line(reconstruction_file)
        total_training_steps += 1
        step += 1

        sample = (total_training_steps % SAMPLE_FREQ == 0)

        output_str = "Epoch: %d, step: %d: "
        output_list = [epoch, step]
        
        #HACK for overfitting
        #line = 'this is a test'

        x = torch.Tensor([fasttext_encode(line, dataset)])

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
            show_text()

        # mean-squared error
        reconstruction_loss = torch.mean((y - decoder_outputs)*(y-decoder_outputs))
   
        loss = 0
        loss += reconstruction_loss
        output_str += "  loss: %.8f"
        output_list.append(reconstruction_loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        track_loss(output_str, output_list)

        if total_training_steps % (10*SAMPLE_FREQ) == 0:
            save_data(output_list, decoder, optimizer)

    print("\n\nTrained epoch: {}, epoch loss: {}\n\n".format(epoch, epoch_loss))

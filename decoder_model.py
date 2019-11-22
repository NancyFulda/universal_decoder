import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import fasttext

import random

#
# ===============================================
#
# GLOBALS

TEACHER_FORCING = True


#
# ===============================================
#
# Helper RNN

class DecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, use_cuda):
        super(DecoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.embedding = nn.Linear(output_size, num_layers*hidden_size)
        self.rnn = nn.GRU(input_size+hidden_size, hidden_size, num_layers=num_layers)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

        self.use_cuda = use_cuda
        if use_cuda:
            self.cuda()

    def forward(self, sentence_embedding, input_var, hidden):
        #self.rnn.flatten_parameters()

        if self.use_cuda:
            input_var = input_var.cuda()

        hidden = hidden.view(self.num_layers, 1, self.hidden_size)

        output = self.embedding(input_var).view(self.num_layers, 1, -1)
        output = F.relu(output)

        output, hidden = self.rnn(torch.cat([sentence_embedding.view([self.num_layers,1,-1]), output],dim=2), hidden)
        output = self.softmax(self.out(output[0]))

        return output, hidden

    def init_hidden_lstm(self):
        result = (Variable(torch.zeros(self.num_layers, 1, self.hidden_size)),
                  Variable(torch.zeros(self.num_layers, 1, self.hidden_size)))
        if self.use_cuda:
            return result.cuda()
        else:
            return result

    def init_hidden_gru(self):
        result = Variable(torch.zeros(self.num_layers, 1, self.hidden_size))
        if self.use_cuda:
            return result.cuda()
        else:
            return result


#
# ===============================================
#
# Main Decoder Model


class Universal_Decoder(nn.Module):

    def __init__(self,
                dataset,
                vocab_dim,
                fasttext_module,
                z_dim,
                decoder_hidden_dim,
                max_length,
                num_layers_for_rnns,
                use_cuda=False):
        super(Universal_Decoder, self).__init__()

        #fastText (for creating output phrases)
        self.ftext = fasttext_module

        # define rnns
        self.num_layers = num_layers_for_rnns
        #!self.decoder_rnn = DecoderRNN(input_size=2*z_dim,
        self.decoder_rnn = DecoderRNN(input_size=z_dim,
                                      hidden_size=decoder_hidden_dim,
                                      output_size=vocab_dim,
                                      num_layers=self.num_layers,
                                      use_cuda=use_cuda)

        # define dense module
        #self.decoder_dense = DecoderDense(z_dim=z_dim,
        #                                  hidden_dim=decoder_hidden_dim)
        self.use_cuda = use_cuda
        self.z_dim = z_dim
        self.max_length = max_length
        self.dataset = dataset

        if use_cuda:
            self.cuda()

    def forward(self, input_variable, target_variable, sample=False):

        # init vars
        target_length = target_variable.shape[0]

        decoder_input = self.dataset.to_onehot([[self.ftext.SOS_index]])
        decoder_input = decoder_input.cuda() if self.use_cuda else decoder_input

        decoder_idxs = np.ones((target_length))
        decoder_outputs = []
        decoder_hidden = self.decoder_rnn.init_hidden_gru()

        rnn_outputs=[]
        for di in range(target_length):
            decoder_output, decoder_hidden = self.decoder_rnn(
                input_variable, decoder_input, decoder_hidden)

            if self.use_cuda:
                decoder_idxs[di] = np.argmax(decoder_output.cpu().data.numpy())
            else:
                decoder_idxs[di] = np.argmax(decoder_output.data.numpy())

            decoder_outputs.append(decoder_output)
            if TEACHER_FORCING and not sample:
                #train using teacher forcing
                decoder_input = target_variable[di]
            else:
                #but sample without it
                one_hot = self.dataset.to_onehot(np.array([decoder_idxs[di]]))
                decoder_input = one_hot
            rnn_outputs.append(decoder_output)

        return torch.stack(decoder_outputs).view([-1,self.dataset.vocab_size]), decoder_idxs, rnn_outputs


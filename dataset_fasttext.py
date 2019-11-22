from collections import Counter
import numpy as np
import codecs
import torch
from torch.autograd import Variable

class Dataset:

    def __init__(self, fasttext_size, max_len=50, min_len=5):
        self.max_len = max_len
        self.min_len = min_len
        self.fasttext_size = fasttext_size
        self.generate_vocab()
        #self.generate_conversations(source_file, target_file)
        #self.create_batches()
        #self.reset_batch_pointer()

    def generate_vocab(self):
        self.vocab_size = self.fasttext_size + 3 #number of fastText vectors + UNK, SOS and EOS
        self.max_sentence_length = 30

    def clean(self, source_line, target_line=None):
        source_line = source_line.strip('\n').strip()
        source_line = source_line.replace('.', ' .')\
                                 .replace(',', ' ,')\
                                 .replace('?', ' ?')\
                                 .replace('!', ' !')\
                                 .replace('"', ' "')\
                                 .replace("'", " '")\
                                 .replace('-', ' -')\
                                 .replace(':', ' :')\
                                 .replace(';', ' ;')
        source_line = source_line.replace('  ',' ').replace('  ', ' ').replace('  ', ' ')
 
        if target_line:       
            target_line = target_line.strip('\n').strip()
            target_line = target_line.replace('.', ' .')\
                                 .replace(',', ' ,')\
                                 .replace('?', ' ?')\
                                 .replace('!', ' !')\
                                 .replace('"', ' "')\
                                 .replace("'", " '")\
                                 .replace('-', ' -')\
                                 .replace(':', ' :')\
                                 .replace(';', ' ;')
            target_line = target_line.replace('  ',' ').replace('  ', ' ').replace('  ', ' ')
            return source_line, target_line

        return source_line
    
    def to_onehot(self, x, long_type=False):
        onehot_stack = torch.zeros((len(x), self.vocab_size))
        onehot_stack[np.array(range(len(x))), x] = 1
        if long_type:
            onehot_stack = onehot_stack.type(torch.LongTensor)
        return Variable(onehot_stack)
    
    def to_n_hot(self, x, long_type=False):
        onehot = torch.zeros(self.vocab_size)
        onehot[x] = 1
        if long_type:
            onehot = onehot.type(torch.LongTensor)
        return Variable(onehot)

    def to_phrase(self, x):
        return "".join([self.chars[x[i]] for i in range(len(x))])
    
    def to_array(self, x):
        #return np.array([self.chars.index(char) for char in x])
        return np.array(x.lower().split(' '))

    #def size(self):
    #    return len(self.encoder_turns)


if __name__ == "__main__":
    dataset = Dataset()
    x, y = dataset.next_batch()

    # print the indicies
    print(x)
    print(y)

    # print the chars
    #print("".join([dataset.chars[x[i]] for i in range(len(x))]))
    #print("".join([dataset.chars[y[i]] for i in range(len(y))]))

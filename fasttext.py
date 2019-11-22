import pickle as pkl
import numpy as np
import sys

class FastText():

    def __init__(self, fasttext_size, sourcefile='word_embeddings/fasttext.en.pkl'):
        with open(sourcefile, 'rb') as myfile:
            data = pkl.load(myfile)
        self.tokens = data['tokens'][:fasttext_size]
        self.vectors = data['vectors'][:fasttext_size]
        self.vocab_size = fasttext_size
        
        #add start- and end-of-sentence markers
        self.tokens.append('UNK')
        self.vectors = np.vstack([self.vectors, np.zeros(300)])
        self.UNK_index = len(self.vectors-1)

        self.tokens.append('SOS')
        self.vectors = np.vstack([self.vectors, -1*np.ones(300)])
        self.SOS_index = len(self.vectors-1)

        self.tokens.append('EOS')
        self.vectors = np.vstack([self.vectors, np.ones(300)])
        self.EOS_index = len(self.vectors-1)

        #maybe useful for optimization?
        self.arr_tokens = np.atleast_2d(self.tokens)

    def get_index(self, word):
        if word in self.tokens:
            return self.tokens.index(word)
        else:
            return self.vocab_size # use the unknown token if we don't recognize the word...

    def get_indices(self, words):
        indices = []
        for w in list(words):
            if w in self.tokens:
                index = self.tokens.index(w)
                indices.append(index)
            else:
                indices.append(self.vocab_size) #use the unknown token if we don't recognize the word...
        return indices
 
        #print("here")
        #print(words)
        #indices = np.atleast_1d([np.squeeze(np.argwhere(self.arr_tokens==w)) for w in words])
        #sys.stdout.flush()
        #return indices[:,1]
    
    def get_words_from_indices(self, indices):
        words = []
        for i in indices:
            words.append(np.squeeze(self.tokens)[i])
        return words
        #indices = np.array([(0,i) for i in list(indices)])
        #print("indices are ", indices)
        #return self.tokens[indices]

    def get_vectors(self, words):
        vectors = []
        for w in words:
            index = list(self.tokens).index(w)
            vectors.append(self.vectors[index])
        return vectors

        #WARNING - code below returns a 3d vector... ?
        #indices = [np.squeeze(np.argwhere(self.tokens==w)) for w in words]
        ##indices = np.array(indices)[:,1]
        #vectors = []
        #for i in indices:
        #    vectors.append(self.vectors[i])
        #return np.array(vectors)

        #indices = np.array([np.squeeze(np.argwhere(self.tokens==w)) for w in words])
        #return self.vectors[indices]

    def get_words(self, vectors):
        words = []
        for v in vectors:
            index = np.squeeze(np.where((self.vectors == np.atleast_2d(v)).all(axis=1)))
            words.append(np.squeeze(self.tokens)[index])
        return words
        #indices = [np.squeeze(np.argwhere((self.vectors==np.atleast_2d(vectors)).all(axis=1))) for w in words]
        #return tokens[indices]

    def n_hot_to_words(self, x):
        indices = np.where(x.data.cpu().numpy()==1)
        output = self.get_words_from_indices(indices)
        return output

    def one_hots_to_words(self, x):
        indices = [np.where(v==1) for v in x.data.cpu().numpy()]
        return self.get_words_from_indices(indices)

if __name__ == "__main__":
    f = FastText()
    text = np.array("I am a purple coconut".lower().split(' '))
    indices = f.get_indices(text)
    print("indices ", indices)
    words = f.get_words_from_indices(indices)
    print("words ", words)
    v=f.get_vectors(text)
    #print(v.shape)
    w = f.get_words(v)
    print(w)
